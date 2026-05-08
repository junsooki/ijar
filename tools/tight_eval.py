"""Tight ISR evaluation on saved best.pt checkpoints with confidence intervals.

Re-evaluates each checkpoint over many episodes to tighten the ISR variance bound,
so we can confidently rank v15 (harmonic) vs v17 (calibrated) vs v16 (no phi).
"""

import argparse
import os
import sys
import math

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SEAM = os.path.dirname(_HERE)
sys.path.insert(0, _SEAM)
sys.path.append(os.path.join(_SEAM, "RAILGUN"))

from models.unet import UNet
from envs.pogema_railgun_env import POGEMARailgunEnv
from tools.heuristic_cost import HeuristicPhiAdapter
from tools.cost_shaping import apply_cost_shaping
from tools.graph_construction import extract_agent_positions


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


def evaluate(ckpt_path, *, density_kernel, phi_alpha, n_episodes, num_agents=4,
             map_size=16, max_steps=128, density=0.3, w_density=2.0, seed=42):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    unet = UNet(n_channels=6, n_classes=5, first_layer_channels=64,
                bilinear=False, blocks_per_stage=0).to(device)
    state = load_checkpoint(ckpt_path, device)
    unet.load_state_dict(state["unet"] if "unet" in state else state, strict=False)
    unet.eval()

    dummy = np.zeros((map_size, map_size), dtype=np.uint8)
    phi_adapter = HeuristicPhiAdapter(
        dummy, w_density=w_density, w_bottleneck=3.0, w_conflict=5.0,
        density_kernel=density_kernel,
    )

    env = POGEMARailgunEnv(num_agents=num_agents, max_steps=max_steps,
                           density=density, size=map_size, seed=seed)

    successes = []  # per-agent reached fraction per episode
    for ep in range(n_episodes):
        feat, _ = env.reset(seed=seed + ep * 7919)
        phi_adapter.update_map(env.obstacle)
        prev_locs = None
        N = env.num_agents
        reached = np.zeros(N, dtype=bool)

        for _ in range(env.max_steps):
            feat_dev = feat.unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = unet(feat_dev)

            positions, agent_ids = extract_agent_positions(feat)
            if prev_locs is None or prev_locs.shape[0] != positions.shape[0]:
                prev_locs = positions.clone()

            if positions.shape[0] == 0:
                feat, rews, done, _ = env.step(np.zeros(N, dtype=np.int64))
                reached |= (rews >= 9.0)
                if done: break
                continue

            phi_costs, _ = phi_adapter(feat, positions, prev_locs, agent_ids)
            ls = logits[0, :, positions[:, 0], positions[:, 1]].T
            sigma = ls.std().clamp(min=0.1).item()
            eff_alpha = phi_alpha / sigma
            shaped = apply_cost_shaping(
                logits, phi_costs.to(device), feat,
                positions.to(device), agent_ids.to(device),
                alpha=eff_alpha, proximity_radius=2,
            )
            ag = shaped[0, :, positions[:, 0], positions[:, 1]].T
            actions_vis = ag.argmax(dim=-1)

            full = np.zeros(N, dtype=np.int64)
            for k_i, a_i in enumerate((agent_ids - 1).long().clamp(0, N-1).tolist()):
                full[a_i] = int(actions_vis[k_i].item())

            prev_locs = positions.clone()
            feat, rews, done, _ = env.step(full)
            reached |= (rews >= 9.0)
            if done: break

        successes.append(reached.sum() / N)

    arr = np.array(successes)
    mean = arr.mean()
    se = arr.std(ddof=1) / math.sqrt(len(arr))
    return mean, se, arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configs = []
    for run, kernel, alpha in [("v16_nophi", "harmonic", 0.0),
                                ("v15_seed2", "harmonic", 1.0),
                                ("v17_calibrated", "calibrated", 1.0)]:
        for it in [50, 100, 150, 200]:
            path = f"runs/{run}/ckpt_{it:05d}.pt"
            if os.path.isfile(path):
                configs.append((f"{run} ckpt{it}", path, kernel, alpha, 2.0))

    print(f"Tight evaluation: {args.episodes} episodes per config, seed={args.seed}")
    print(f"{'config':<50} {'ISR':>8} {'± SE':>8} {'95% CI':>20}")
    print("-" * 90)
    for name, path, kernel, alpha, wd in configs:
        if not os.path.isfile(path):
            print(f"{name:<50}  (checkpoint missing: {path})")
            continue
        mean, se, _ = evaluate(path, density_kernel=kernel, phi_alpha=alpha,
                                n_episodes=args.episodes, w_density=wd, seed=args.seed)
        ci_lo = mean - 1.96 * se
        ci_hi = mean + 1.96 * se
        print(f"{name:<50} {mean:>8.4f} {se:>8.4f}  [{ci_lo:.3f}, {ci_hi:.3f}]")


if __name__ == "__main__":
    main()
