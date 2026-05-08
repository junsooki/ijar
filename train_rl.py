"""SEAM — Social-cost Enhancement for Agent Movement.

PPO fine-tuning of the pretrained RAILGUN UNet with heuristic cooperative
cost shaping (density + betweenness + directional conflict).

Usage:
    python train_rl.py --config configs/rl_ppo.yaml
    python train_rl.py --unet_checkpoint results/checkpoints/railgun_pretrained.pt \\
                       --map_source RAILGUN/data/map_files/maze-32-32-10-4-75 \\
                       --num_agents 4

Architecture:
  - Policy: UNet [6,H,W] → shaped logits [5,H,W] → per-agent action
  - Value:  global-average-pooled logits → 2-layer MLP → scalar V(s)
  - Parameter sharing: all N agents share the same UNet + value head

Requires RAILGUN to be installed/cloned at RAILGUN/ and importable.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

# seam/ must be first so our tools/ package takes precedence over RAILGUN/tools/
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
sys.path.append(os.path.join(_HERE, "RAILGUN"))  # RAILGUN at end for models/

from models.unet import UNet

# CRAFT imports
from envs.pogema_railgun_env import POGEMARailgunEnv
from tools.heuristic_cost import HeuristicPhiAdapter, precompute_betweenness
from tools.audit import AuditLogger, collect_phi_stats, collect_action_stats

from tools.cost_shaping import apply_cost_shaping
from tools.graph_construction import extract_agent_positions

# RAILGUN action deltas: 0=stay, 1=right, 2=left, 3=up, 4=down
_ACTION_DELTAS = [(0, 0), (0, 1), (0, -1), (-1, 0), (1, 0)]


def compute_action_mask(feat: torch.Tensor, positions: torch.Tensor,
                         agent_ids: torch.Tensor, N: int) -> torch.Tensor:
    """[N, 5] bool mask: True=valid. Stays always valid; movements invalid if
    target cell is an obstacle or off-grid. Invisible agents → all-stay mask."""
    H, W = feat.shape[-2:]
    obstacle = feat[0]  # [H, W], 1=wall
    mask = torch.zeros(N, 5, dtype=torch.bool)
    mask[:, 0] = True  # stay always valid
    if positions.shape[0] == 0:
        return mask
    for k in range(positions.shape[0]):
        r = int(positions[k, 0].item())
        c = int(positions[k, 1].item())
        a_idx = int(agent_ids[k].item()) - 1
        if not (0 <= a_idx < N):
            continue
        for a in range(1, 5):
            dr, dc = _ACTION_DELTAS[a]
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and obstacle[nr, nc].item() < 0.5:
                mask[a_idx, a] = True
    return mask


# ──────────────────────────────────────────────────────────────────────────────
# Value head
# ──────────────────────────────────────────────────────────────────────────────

class ValueHead(nn.Module):
    """Global-average-pool over logits → scalar state value V(s).

    Input: logits [B, 5, H, W] (UNet output before softmax)
    Output: [B] scalar value estimate
    """
    def __init__(self, action_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # [B, 5, H, W] → [B, 5] via global average pool
        pooled = logits.mean(dim=(-2, -1))  # [B, 5]
        return self.net(pooled).squeeze(-1)  # [B]


# ──────────────────────────────────────────────────────────────────────────────
# Rollout collection
# ──────────────────────────────────────────────────────────────────────────────

def collect_rollout(
    env: POGEMARailgunEnv,
    unet: UNet,
    value_head: ValueHead,
    phi_adapter: HeuristicPhiAdapter,
    device: torch.device,
    rollout_steps: int,
    phi_alpha: float,
    phi_proximity_radius: int,
    reward_scale: float = 1.0,
    cfg_action_masking: bool = False,
    policy_aware_phi: bool = False,
) -> dict:
    """Collect a rollout of `rollout_steps` environment steps.

    Returns a dict with tensors:
      states      [T, 6, H, W]
      actions     [T, N]         int64 — RAILGUN action indices
      log_probs   [T, N]         float32
      rewards     [T, N]         float32
      values      [T]            float32  (shared state value)
      dones       [T]            bool
      audit       dict           phi stats + action stats for this rollout
    """
    N = env.num_agents

    states_list        = []
    actions_list       = []
    log_probs_list     = []
    rewards_list       = []
    values_list        = []
    dones_list         = []
    phi_costs_list     = []   # audit
    phi_comp_list      = []   # audit
    positions_list     = []   # audit
    shaped_logits_list = []   # fix1: store shaped agent logits for PPO reference
    shaping_delta_list = []   # fix1: shaped - raw per-agent logits
    logit_scales_list  = []   # fix2: track logit scale per step

    feat, _ = env.reset()
    phi_adapter.update_map(env.obstacle)
    prev_locs = None

    unet.eval()
    value_head.eval()

    with torch.no_grad():
        for _ in range(rollout_steps):
            feat_dev = feat.unsqueeze(0).to(device)  # [1, 6, H, W]

            # UNet forward
            logits, _ = unet(feat_dev)  # [1, 5, H, W]

            # Extract visible agent positions (may be < N if some agents finished)
            positions, agent_ids = extract_agent_positions(feat)
            if prev_locs is None or prev_locs.shape[0] != positions.shape[0]:
                prev_locs = positions.clone()

            phi_costs, phi_components = phi_adapter(feat, positions, prev_locs, agent_ids)
            phi_costs_dev = phi_costs.to(device)
            positions_dev = positions.to(device)
            agent_ids_dev = agent_ids.to(device)

            # FIX 2: normalize phi penalty by logit scale
            if positions.shape[0] > 0:
                raw_agent_logits_for_scale = logits[0, :, positions[:, 0], positions[:, 1]].T
                logit_scale = raw_agent_logits_for_scale.std().clamp(min=0.1).item()
            else:
                logit_scale = 1.0
            effective_alpha = phi_alpha / logit_scale

            shaped = apply_cost_shaping(
                logits, phi_costs_dev, feat, positions_dev, agent_ids_dev,
                alpha=effective_alpha, proximity_radius=phi_proximity_radius,
                policy_aware=policy_aware_phi,
            )  # [1, 5, H, W]

            # Build full [N, 5] logit matrices — stay (uniform) for invisible agents
            full_shaped_logits = torch.zeros(N, 5, device=device)
            full_raw_logits    = torch.zeros(N, 5, device=device)
            vis_idx = (agent_ids - 1).long()  # 0-indexed agent slots for visible agents
            if positions.shape[0] > 0:
                vis_shaped = shaped[0, :, positions[:, 0], positions[:, 1]].T   # [k, 5]
                vis_raw    = logits[0, :, positions[:, 0], positions[:, 1]].T   # [k, 5]
                for k_i, a_i in enumerate(vis_idx.clamp(0, N-1).tolist()):
                    full_shaped_logits[a_i] = vis_shaped[k_i]
                    full_raw_logits[a_i]    = vis_raw[k_i]

            # ACTION MASKING (optional): zero out probability for moves into obstacles
            if cfg_action_masking:
                action_mask = compute_action_mask(feat, positions, agent_ids, N).to(device)
                full_shaped_logits = full_shaped_logits.masked_fill(~action_mask, -1e8)

            # FIX 1: compute and store shaped logits and shaping delta
            shaping_delta = full_shaped_logits - full_raw_logits         # [N, 5]
            shaped_logits_list.append(full_shaped_logits.cpu())
            shaping_delta_list.append(shaping_delta.cpu())
            logit_scales_list.append(logit_scale)

            dist = torch.distributions.Categorical(logits=full_shaped_logits)
            actions   = dist.sample()        # [N]
            log_probs = dist.log_prob(actions)  # [N]

            # State value (shared)
            value = value_head(logits)       # [1] → scalar

            # Step environment — always pass N actions
            prev_locs = positions.clone()
            feat_next, rewards, done, _ = env.step(actions.cpu().numpy())

            # Store
            states_list.append(feat.cpu())
            actions_list.append(actions.cpu())
            log_probs_list.append(log_probs.cpu())
            rewards_list.append(torch.from_numpy(rewards) * reward_scale)
            values_list.append(value.cpu().squeeze(0))
            dones_list.append(torch.tensor(done, dtype=torch.bool))
            phi_costs_list.append(phi_costs.cpu())
            phi_comp_list.append({k: v.cpu() for k, v in phi_components.items()})
            positions_list.append(positions.cpu())

            if done:
                feat, _ = env.reset()
                phi_adapter.update_map(env.obstacle)
                prev_locs = None
            else:
                feat = feat_next

    phi_stats = collect_phi_stats(phi_costs_list, phi_comp_list)
    phi_stats["mean_logit_scale"] = float(np.mean(logit_scales_list))  # fix2: audit
    audit = {
        "phi":     phi_stats,
        "actions": collect_action_stats(actions_list, positions_list),
    }

    return {
        "states":              torch.stack(states_list),        # [T, 6, H, W]
        "actions":             torch.stack(actions_list),       # [T, N]
        "log_probs":           torch.stack(log_probs_list),     # [T, N]
        "rewards":             torch.stack(rewards_list),       # [T, N]
        "values":              torch.stack(values_list),        # [T]
        "dones":               torch.stack(dones_list),         # [T]
        "shaped_agent_logits": torch.stack(shaped_logits_list), # [T, N, 5]  fix1
        "shaping_delta":       torch.stack(shaping_delta_list), # [T, N, 5]  fix1
        "audit":               audit,
    }


# ──────────────────────────────────────────────────────────────────────────────
# GAE + returns
# ──────────────────────────────────────────────────────────────────────────────

def compute_gae(
    rewards: torch.Tensor,  # [T, N]
    values: torch.Tensor,   # [T]
    dones: torch.Tensor,    # [T]
    gamma: float,
    gae_lambda: float,
    last_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (advantages [T, N], returns [T])."""
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(N)

    next_value = last_value
    for t in reversed(range(T)):
        mask = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae
        next_value = values[t].item()

    returns = advantages + values.unsqueeze(1).expand_as(advantages)
    return advantages, returns


class RunningStat:
    """Welford-style running mean/std with EMA decay.

    Used to normalize returns across rollouts so the value head sees a stable
    target distribution (per-batch normalization makes the target shift every
    update, preventing learning).
    """
    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self.mean = 0.0
        self.var = 1.0
        self.initialized = False

    def update(self, x: torch.Tensor) -> None:
        batch_mean = x.mean().item()
        batch_var = x.var(unbiased=False).item()
        if not self.initialized:
            self.mean = batch_mean
            self.var = batch_var
            self.initialized = True
        else:
            self.mean = self.decay * self.mean + (1 - self.decay) * batch_mean
            self.var = self.decay * self.var + (1 - self.decay) * batch_var

    @property
    def std(self) -> float:
        return float(self.var) ** 0.5


# ──────────────────────────────────────────────────────────────────────────────
# PPO update
# ──────────────────────────────────────────────────────────────────────────────

def ppo_update(
    unet: UNet,
    value_head: ValueHead,
    phi_adapter: HeuristicPhiAdapter,
    optimizer: torch.optim.Optimizer,
    rollout: dict,
    device: torch.device,
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
    ppo_epochs: int,
    minibatch_size: int,
    phi_alpha: float,
    phi_proximity_radius: int,
    gamma: float,
    gae_lambda: float,
    return_stats: "RunningStat | None" = None,
) -> dict:
    """Run PPO update on collected rollout. Returns dict of loss metrics."""
    T, N = rollout["actions"].shape

    # Compute GAE with no-gradient last value estimate
    unet.eval(); value_head.eval()
    with torch.no_grad():
        last_feat = rollout["states"][-1].unsqueeze(0).to(device)
        last_logits, _ = unet(last_feat)
        last_val = value_head(last_logits).item()

    advantages, returns = compute_gae(
        rollout["rewards"], rollout["values"], rollout["dones"],
        gamma, gae_lambda, last_val,
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    # Two return-normalization strategies:
    # - return_stats provided: running EMA stats (more principled but unstable in practice)
    # - return_stats is the string "perbatch": v2-style per-batch normalization (stable
    #   but locks v_loss at 1.0 since targets shift every update)
    if return_stats == "perbatch":
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    elif return_stats is not None:
        return_stats.update(returns)
        returns = (returns - return_stats.mean) / (return_stats.std + 1e-8)

    # Flatten: treat each (step, agent) as an independent sample
    flat_states         = rollout["states"].unsqueeze(1).expand(-1, N, -1, -1, -1).reshape(T * N, *rollout["states"].shape[1:])
    flat_actions        = rollout["actions"].reshape(T * N)
    flat_old_lp         = rollout["log_probs"].reshape(T * N)   # kept for reference/audit
    flat_adv            = advantages.reshape(T * N)
    flat_returns        = returns.reshape(T * N)
    flat_agent_idx      = torch.arange(N).repeat(T)             # which agent (0..N-1) at each sample
    flat_old_shaped     = rollout["shaped_agent_logits"].reshape(T * N, 5)  # fix1: old shaped logits
    flat_shaping_delta  = rollout["shaping_delta"].reshape(T * N, 5)        # fix1: phi correction

    # Also store positions per step — needed for cost shaping during update
    # We re-extract positions from states on-the-fly (cheaper than storing them)

    unet.train()
    # Freeze BatchNorm running stats — standard practice for fine-tuning pretrained models.
    # Keeps weights trainable but prevents stats from drifting on small batches.
    for m in unet.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
    value_head.train()

    total_samples = T * N
    indices = torch.randperm(total_samples)

    metrics = {
        "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
        "clip_frac": 0.0, "approx_kl": 0.0, "grad_norm": 0.0,
        "n_updates": 0,
    }

    for epoch in range(ppo_epochs):
        for start in range(0, total_samples, minibatch_size):
            mb_idx = indices[start:start + minibatch_size]
            mb_states        = flat_states[mb_idx].to(device)
            mb_actions       = flat_actions[mb_idx].to(device)
            mb_adv           = flat_adv[mb_idx].to(device)
            mb_returns       = flat_returns[mb_idx].to(device)
            mb_ag_idx        = flat_agent_idx[mb_idx]
            mb_shaping_delta = flat_shaping_delta[mb_idx].to(device)   # fix1: [B, 5]
            mb_old_shaped    = flat_old_shaped[mb_idx].to(device)       # fix1: [B, 5]

            # Forward pass
            logits, _ = unet(mb_states)          # [B, 5, H, W]
            values_pred = value_head(logits)     # [B]

            # Extract per-agent raw logits (need positions — re-read from state ch1)
            B = mb_states.size(0)
            agent_logits = torch.zeros(B, 5, device=device)
            for b in range(B):
                own_id = mb_ag_idx[b].item() + 1  # 1-based
                ch1 = mb_states[b, 1]             # [H, W]
                pos = (ch1 == own_id).nonzero(as_tuple=False)
                if pos.size(0) > 0:
                    r, c = pos[0, 0], pos[0, 1]
                    agent_logits[b] = logits[b, :, r, c]
                else:
                    agent_logits[b] = logits[b].mean(dim=(-2, -1))

            # FIX 1: apply stored shaping delta so new logits match the same
            # distribution family (shaped) as the old logits collected during rollout
            shaped_agent_logits = agent_logits + mb_shaping_delta   # [B, 5]
            dist = torch.distributions.Categorical(logits=shaped_agent_logits)
            new_log_probs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()

            # FIX 1: recompute old log-probs from stored shaped logits so that
            # both old and new come from the same distribution type
            old_dist = torch.distributions.Categorical(logits=mb_old_shaped)
            mb_old_lp_shaped = old_dist.log_prob(mb_actions)

            # PPO clipped surrogate
            ratio = (new_log_probs - mb_old_lp_shaped).exp()
            surr1 = ratio * mb_adv
            surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values_pred, mb_returns)

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            trainable_params = [p for p in unet.parameters() if p.requires_grad] + list(value_head.parameters())
            grad_norm = nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5).item()
            optimizer.step()

            # Audit metrics
            with torch.no_grad():
                clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()
                approx_kl = (mb_old_lp_shaped - new_log_probs).mean().item()  # fix1: use shaped ref

            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"]  += value_loss.item()
            metrics["entropy"]     += entropy.item()
            metrics["clip_frac"]   += clip_frac
            metrics["approx_kl"]   += approx_kl
            metrics["grad_norm"]   += grad_norm
            metrics["n_updates"]   += 1

    n = metrics["n_updates"]
    for k in ("policy_loss", "value_loss", "entropy", "clip_frac", "approx_kl", "grad_norm"):
        metrics[k] /= max(n, 1)
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# ISR evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_isr(
    env: POGEMARailgunEnv,
    unet: UNet,
    value_head: ValueHead,
    phi_adapter: HeuristicPhiAdapter,
    device: torch.device,
    n_episodes: int,
    phi_alpha: float,
    phi_proximity_radius: int,
    policy_aware_phi: bool = False,
) -> float:
    """Individual Success Rate: fraction of agents reaching their goal."""
    unet.eval(); value_head.eval()
    total_agents = 0
    total_reached = 0

    for ep in range(n_episodes):
        feat, _ = env.reset()
        phi_adapter.update_map(env.obstacle)
        prev_locs = None
        N = env.num_agents
        reached = np.zeros(N, dtype=bool)

        for _ in range(env.max_steps):
            feat_dev = feat.unsqueeze(0).to(device)
            logits, _ = unet(feat_dev)

            positions, agent_ids = extract_agent_positions(feat)
            if prev_locs is None or prev_locs.shape[0] != positions.shape[0]:
                prev_locs = positions.clone()

            # If all agents have reached their goals, just step with stay actions
            if positions.shape[0] == 0:
                feat, rews, done, info = env.step(np.zeros(env.num_agents, dtype=np.int64))
                reached |= (rews >= 9.0)
                if done:
                    break
                continue

            phi_costs, _ = phi_adapter(feat, positions, prev_locs, agent_ids)
            # FIX 2: normalize phi penalty by logit scale (consistent with collect_rollout)
            raw_agent_logits_for_scale = logits[0, :, positions[:, 0], positions[:, 1]].T  # [N, 5]
            logit_scale = raw_agent_logits_for_scale.std().clamp(min=0.1).item()
            effective_alpha = phi_alpha / logit_scale
            shaped = apply_cost_shaping(
                logits, phi_costs.to(device), feat,
                positions.to(device), agent_ids.to(device),
                alpha=effective_alpha, proximity_radius=phi_proximity_radius,
                policy_aware=policy_aware_phi,
            )

            agent_logits = shaped[0, :, positions[:, 0], positions[:, 1]].T  # [k, 5]
            actions_vis = agent_logits.argmax(dim=-1)  # greedy for eval, [k]

            # POGEMA requires len(action)==num_agents always. Map visible-agent
            # actions back to the full N slots; default to stay (0) for agents
            # that have already reached their goal (no longer in feat ch1).
            full_actions = np.zeros(env.num_agents, dtype=np.int64)
            vis_idx = (agent_ids - 1).long().clamp(0, env.num_agents - 1)
            for k_i, a_i in enumerate(vis_idx.tolist()):
                full_actions[a_i] = int(actions_vis[k_i].item())

            prev_locs = positions.clone()
            feat, rews, done, info = env.step(full_actions)

            # Use reward threshold to detect goal-reach (reward = -0.1 + 10.0 = 9.9)
            reached |= (rews >= 9.0)
            if done:
                break

        total_agents  += N
        total_reached += reached.sum()

    return total_reached / max(total_agents, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description="CRAFT PPO training")
    parser.add_argument("--config", type=str, default=None)
    # Overrideable at CLI
    parser.add_argument("--unet_checkpoint", type=str, default=None)
    parser.add_argument("--map_source", type=str, default=None)
    parser.add_argument("--num_agents", type=int, default=None)
    parser.add_argument("--map_size", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    # Load YAML config as base
    cfg = {}
    if args.config and os.path.isfile(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    # CLI overrides
    for key in ("unet_checkpoint", "map_source", "num_agents", "map_size",
                "max_iterations", "run_name"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val

    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    cfg = get_args()

    # ── Device ──────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Logging ──────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.get("run_name") or f"craft_ppo_{timestamp}"
    log_dir = os.path.join(cfg.get("log_dir", "runs"), run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    audit_interval = cfg.get("audit_interval", 10)
    auditor = AuditLogger(log_dir, writer, print_interval=audit_interval)
    # Save config
    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)
    print(f"Logging to {log_dir}")

    # ── Curriculum ───────────────────────────────────────────────────────────
    curriculum = cfg.get("curriculum", [
        {"num_agents": cfg.get("num_agents", 4),
         "map_size": cfg.get("map_size", 16),
         "max_steps": cfg.get("max_steps_per_episode", 128),
         "isr_threshold": None},
    ])
    curr_stage = 0

    def make_env(stage_cfg):
        return POGEMARailgunEnv(
            map_source=cfg.get("map_source"),
            num_agents=stage_cfg["num_agents"],
            max_steps=stage_cfg["max_steps"],
            density=cfg.get("map_density", 0.3),
            size=stage_cfg.get("map_size", 16),
        )

    env       = make_env(curriculum[curr_stage])
    eval_env  = make_env(curriculum[curr_stage])

    # ── Models ───────────────────────────────────────────────────────────────
    unet = UNet(
        n_channels=cfg.get("feature_dim", 6),
        n_classes=cfg.get("action_dim", 5),
        first_layer_channels=cfg.get("first_layer_channels", 64),
        bilinear=cfg.get("bilinear", True),
        blocks_per_stage=cfg.get("blocks_per_stage", 0),
    ).to(device)

    def _load_checkpoint(path):
        """Load checkpoint, handling both plain UNet dicts and full SEAM checkpoints."""
        try:
            return torch.load(path, map_location=device, weights_only=True)
        except Exception:
            return torch.load(path, map_location=device, weights_only=False)

    ckpt_path = cfg.get("unet_checkpoint")
    if ckpt_path and os.path.isfile(ckpt_path):
        state = _load_checkpoint(ckpt_path)
        if "unet" in state:
            unet.load_state_dict(state["unet"], strict=False)
            print(f"Loaded UNet from SEAM checkpoint {ckpt_path}")
        else:
            unet.load_state_dict(state, strict=False)
            print(f"Loaded UNet from {ckpt_path}")
    else:
        print("WARNING: No UNet checkpoint — training from random init.")

    # Freeze most of the UNet; only fine-tune the last decoder block + output head.
    # This gives enough capacity to learn cooperative behavior without catastrophic
    # gradient explosion through 31M params.
    freeze_backbone = cfg.get("freeze_backbone", True)
    if freeze_backbone:
        # Keep up4 (last decoder block) and output_conv trainable; freeze everything else
        trainable_layers = cfg.get("trainable_layers", ["up4", "output_conv"])
        for name, param in unet.named_parameters():
            if not any(tl in name for tl in trainable_layers):
                param.requires_grad_(False)
        n_trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        print(f"Backbone partially frozen. Trainable UNet params: {n_trainable:,} ({', '.join(trainable_layers)})")

    value_head = ValueHead(action_dim=cfg.get("action_dim", 5)).to(device)
    if ckpt_path and os.path.isfile(ckpt_path) and cfg.get("load_value_head", False):
        state_vh = _load_checkpoint(ckpt_path)
        if "value_head" in state_vh:
            value_head.load_state_dict(state_vh["value_head"])
            print(f"Loaded value head from SEAM checkpoint")

    # ── Heuristic phi adapter ─────────────────────────────────────────────
    # Initialise with a dummy map; update_map() is called at each episode reset
    dummy_obs = np.zeros((cfg.get("map_size", 16), cfg.get("map_size", 16)), dtype=np.uint8)
    phi_adapter = HeuristicPhiAdapter(
        dummy_obs,
        w_density=cfg.get("phi_w_density", 2.0),
        w_bottleneck=cfg.get("phi_w_bottleneck", 3.0),
        w_conflict=cfg.get("phi_w_conflict", 5.0),
        density_kernel=cfg.get("density_kernel", "harmonic"),
    )

    # ── Optimizer ────────────────────────────────────────────────────────────
    trainable = [p for p in unet.parameters() if p.requires_grad] + list(value_head.parameters())
    optimizer = torch.optim.Adam(trainable, lr=cfg.get("learning_rate", 3e-5))

    # ── Training loop ────────────────────────────────────────────────────────
    max_iter  = cfg.get("max_iterations", 2000)
    eval_int  = cfg.get("eval_interval", 50)
    save_int  = cfg.get("save_interval", 100)
    best_isr  = 0.0

    phi_alpha = cfg.get("phi_alpha", 1.0)
    phi_prox  = cfg.get("phi_proximity_radius", 2)

    # Return normalization strategy:
    # - "perbatch" (v2-style): stable, but value head can't learn variance patterns
    # - "running" (EMA): value head learns, but high PPO clip rates from drift
    # - None (raw): only works if reward_scale keeps returns bounded
    norm_mode = cfg.get("return_norm", None)  # "perbatch" | "running" | None
    if norm_mode == "running":
        return_stats = RunningStat(decay=0.99)
    elif norm_mode == "perbatch":
        return_stats = "perbatch"
    else:
        return_stats = None

    # Entropy schedule: linear decay from entropy_coef_start to entropy_coef_end over decay_iters
    ent_start = cfg.get("entropy_coef", 0.01)
    ent_end   = cfg.get("entropy_coef_end", ent_start)
    ent_decay_iters = cfg.get("entropy_decay_iters", max_iter)

    for iteration in range(1, max_iter + 1):
        t0 = time.time()

        # Linear entropy decay
        frac = min(1.0, (iteration - 1) / max(ent_decay_iters, 1))
        entropy_coef_now = ent_start + (ent_end - ent_start) * frac

        # Collect rollout
        rollout = collect_rollout(
            env, unet, value_head, phi_adapter, device,
            rollout_steps=cfg.get("rollout_steps", 512),
            phi_alpha=phi_alpha, phi_proximity_radius=phi_prox,
            reward_scale=cfg.get("reward_scale", 1.0),
            cfg_action_masking=cfg.get("action_masking", False),
            policy_aware_phi=cfg.get("policy_aware_phi", False),
        )

        # PPO update
        metrics = ppo_update(
            unet, value_head, phi_adapter, optimizer, rollout, device,
            clip_eps=cfg.get("clip_eps", 0.2),
            entropy_coef=entropy_coef_now,
            value_coef=cfg.get("value_coef", 0.5),
            ppo_epochs=cfg.get("ppo_epochs", 4),
            minibatch_size=cfg.get("minibatch_size", 256),
            phi_alpha=phi_alpha, phi_proximity_radius=phi_prox,
            gamma=cfg.get("gamma", 0.99),
            gae_lambda=cfg.get("gae_lambda", 0.95),
            return_stats=return_stats,
        )

        elapsed = time.time() - t0
        mean_reward = rollout["rewards"].mean().item()

        writer.add_scalar("Train/policy_loss",  metrics["policy_loss"],  iteration)
        writer.add_scalar("Train/value_loss",   metrics["value_loss"],   iteration)
        writer.add_scalar("Train/entropy",      metrics["entropy"],      iteration)
        writer.add_scalar("Train/mean_reward",  mean_reward,             iteration)
        writer.add_scalar("Train/iter_time_s",  elapsed,                 iteration)
        writer.add_scalar("Audit/logit_scale",  rollout["audit"]["phi"].get("mean_logit_scale", 1.0), iteration)  # fix2
        writer.add_scalar("Train/entropy_coef", entropy_coef_now, iteration)
        if isinstance(return_stats, RunningStat):
            writer.add_scalar("Train/return_mean", return_stats.mean, iteration)
            writer.add_scalar("Train/return_std",  return_stats.std,  iteration)

        print(f"[{iteration:4d}/{max_iter}] "
              f"rew={mean_reward:.3f}  "
              f"π_loss={metrics['policy_loss']:.4f}  "
              f"v_loss={metrics['value_loss']:.4f}  "
              f"ent={metrics['entropy']:.4f}  "
              f"({elapsed:.1f}s)")

        auditor.record(iteration, rollout["audit"], metrics,
                       mean_reward=mean_reward)

        # ── Evaluation ───────────────────────────────────────────────────────
        if iteration % eval_int == 0:
            isr = evaluate_isr(
                eval_env, unet, value_head, phi_adapter, device,
                n_episodes=cfg.get("eval_episodes", 20),
                phi_alpha=phi_alpha, phi_proximity_radius=phi_prox,
                policy_aware_phi=cfg.get("policy_aware_phi", False),
            )
            writer.add_scalar("Eval/ISR", isr, iteration)
            print(f"  → ISR = {isr:.3f}  (best = {best_isr:.3f})")
            # Also append a small ISR-only audit entry so plot_curves.py can
            # read isr per checkpoint without parsing stdout.
            auditor._log_jsonl(iteration, {}, {}, extra={"isr": float(isr)})

            if isr > best_isr:
                best_isr = isr
                torch.save({
                    "unet": unet.state_dict(),
                    "value_head": value_head.state_dict(),
                    "iteration": int(iteration),
                    "isr": float(isr),
                }, os.path.join(log_dir, "best.pt"))

            # ── Curriculum advancement ─────────────────────────────────────
            stage_thr = curriculum[curr_stage].get("isr_threshold")
            if stage_thr is not None and isr >= stage_thr and curr_stage + 1 < len(curriculum):
                curr_stage += 1
                new_stage = curriculum[curr_stage]
                print(f"  *** Curriculum: advancing to stage {curr_stage}: "
                      f"{new_stage['num_agents']} agents, {new_stage['map_size']}x{new_stage['map_size']} ***")
                env      = make_env(new_stage)
                eval_env = make_env(new_stage)
                writer.add_scalar("Train/curriculum_stage", curr_stage, iteration)

        # ── Checkpoint ───────────────────────────────────────────────────────
        if iteration % save_int == 0:
            torch.save({
                "unet": unet.state_dict(),
                "value_head": value_head.state_dict(),
                "iteration": int(iteration),
            }, os.path.join(log_dir, f"ckpt_{iteration:05d}.pt"))

    # Final checkpoint
    torch.save({
        "unet": unet.state_dict(),
        "value_head": value_head.state_dict(),
        "iteration": int(max_iter),
    }, os.path.join(log_dir, "final.pt"))
    print(f"\nTraining complete. Best ISR: {best_isr:.3f}")
    auditor.close()
    writer.close()


if __name__ == "__main__":
    main()
