"""Iter 18 tight eval: pretrained vs PPO-trained checkpoint on POGEMA random 4ag/16x16.

Adapts diag_l_tight_compare.py from maze-distribution eval to POGEMA-random eval.
Supports MoEPolicy checkpoints (auto-detected by presence of 'gate.*' / 'new_out.*' keys).

Usage:
  PPO_BEST=runs/ppo_v18a_moe/best.pt \
    python tools/diag_pogema_compare.py --episodes 200 --num_agents 4 --map_size 16 \
                                         --density 0.3 --max_steps 128 --seed 42
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SEAM = os.path.dirname(_HERE)
sys.path.insert(0, _SEAM)
sys.path.append(os.path.join(_SEAM, "RAILGUN"))

from models.unet import UNet
from envs.pogema_railgun_env import POGEMARailgunEnv
from tools.graph_construction import extract_agent_positions
def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)
# RAILGUN canonical action deltas: 0=stay, 1=right, 2=left, 3=up, 4=down
ACTION_DELTAS = torch.tensor(
    [[0, 0], [0, 1], [0, -1], [-1, 0], [1, 0]],
    dtype=torch.long,
)


def load_model_from_ckpt(ckpt_path, device):
    """Load either a plain UNet or an MoEPolicy checkpoint.

    Pretrained checkpoints store just UNet weights. PPO-trained MoE checkpoints
    save the full MoEPolicy state_dict (with 'gate.*' and 'new_out.*' submodules).
    """
    state = load_checkpoint(ckpt_path, device)
    sd = state["unet"] if isinstance(state, dict) and "unet" in state else state
    keys = list(sd.keys()) if isinstance(sd, dict) else []
    is_moe = any("gate." in k or "new_out." in k for k in keys)
    is_gnn = any(k.startswith("gnn.") for k in keys)
    is_refinement = any(k.startswith("feedback_encoder.") for k in keys)
    if is_refinement:
        from tools.iterative_refinement import IterativeRefinementPolicy
        u = UNet(n_channels=6, n_classes=5, first_layer_channels=64,
                 blocks_per_stage=0, bilinear=False).to(device)
        ref = IterativeRefinementPolicy(u).to(device)
        ref.load_state_dict(sd, strict=False)
        ref.eval()
        return ref
    if is_moe:
        from train_rl import MoEPolicy
        u = UNet(n_channels=6, n_classes=5, first_layer_channels=64,
                 blocks_per_stage=0, bilinear=False).to(device)
        moe = MoEPolicy(u, in_channels=6, n_classes=5).to(device)
        moe.load_state_dict(sd, strict=False)
        moe.eval()
        return moe
    if is_gnn:
        from train_rl import GNNPolicy
        u = UNet(n_channels=6, n_classes=5, first_layer_channels=64,
                 blocks_per_stage=0, bilinear=False).to(device)
        # Detect variant from key set
        variant = "magat_plus" if any("edge_mlp" in k or "magat" in k for k in keys) else "naive"
        # Detect num_rounds from how many GNN layer indices present
        layer_indices = set()
        for k in keys:
            if k.startswith("gnn.layers."):
                try:
                    layer_indices.add(int(k.split(".")[2]))
                except ValueError:
                    pass
        num_rounds = max(layer_indices) + 1 if layer_indices else 3
        gnn = GNNPolicy(u, max_agents=32,
                        gnn_kwargs={"num_heads": 4, "num_rounds": num_rounds},
                        gnn_variant=variant, warm_start=True).to(device)
        gnn.load_state_dict(sd, strict=False)
        gnn.eval()
        return gnn
    u = UNet(n_channels=6, n_classes=5, first_layer_channels=64,
             blocks_per_stage=0, bilinear=False).to(device)
    u.load_state_dict(sd, strict=False)
    u.eval()
    return u


def evaluate(ckpt_path, *, n_episodes, num_agents, map_size, density, max_steps, seed):
    if os.environ.get("SEAM_FORCE_CPU") == "1":
        device = torch.device("cpu")
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model_from_ckpt(ckpt_path, device)

    # POGEMA random maps: pass map_source=None and provide size/density.
    env = POGEMARailgunEnv(map_source=None, num_agents=num_agents,
                           max_steps=max_steps, seed=seed,
                           density=density, size=map_size)
    deltas = ACTION_DELTAS

    successes = []
    for ep in range(n_episodes):
        feat, _ = env.reset(seed=seed + ep * 7919)
        N = env.num_agents
        H, W = env.map_shape
        reached = np.zeros(N, dtype=bool)

        # Detect refinement model so we can run adaptive K-pass inference.
        from tools.iterative_refinement import IterativeRefinementPolicy
        from tools.conflict_features import compute_conflict_features
        _is_ref = isinstance(model, IterativeRefinementPolicy)
        _kmax = int(os.environ.get("REFINEMENT_KMAX", "3")) if _is_ref else 1

        for _ in range(env.max_steps):
            with torch.no_grad():
                feat_dev = feat.unsqueeze(0).to(device)
                if _is_ref:
                    # Iterative refinement loop with early stop on no-conflict.
                    A_prev = None; C_prev = None
                    logits = None
                    for _k in range(_kmax):
                        logits, _ = model(feat_dev, prev_action_map=A_prev, prev_conflict_map=C_prev)
                        # Build action map + conflict features for next pass / early-stop.
                        positions_k, agent_ids_k = extract_agent_positions(feat)
                        if positions_k.shape[0] == 0:
                            break
                        ag_logits_k = logits[0, :, positions_k[:, 0], positions_k[:, 1]].T  # [K, 5]
                        # Apply action mask before argmax (same as final selection).
                        deltas_k = ACTION_DELTAS
                        next_pos_k = positions_k.unsqueeze(1) + deltas_k.unsqueeze(0)
                        nr_k = next_pos_k[:, :, 0]; nc_k = next_pos_k[:, :, 1]
                        H_k, W_k = feat.shape[-2:]
                        in_bounds_k = (nr_k >= 0) & (nr_k < H_k) & (nc_k >= 0) & (nc_k < W_k)
                        valid_k = in_bounds_k.clone()
                        if in_bounds_k.any():
                            valid_k[in_bounds_k] = feat[0][nr_k[in_bounds_k], nc_k[in_bounds_k]] == 0
                        valid_k[:, 0] = True
                        ag_logits_k_masked = ag_logits_k.masked_fill(~valid_k.to(device), float("-inf"))
                        actions_k = ag_logits_k_masked.argmax(dim=-1)  # [K]
                        # Build action map
                        A_prev = torch.zeros(1, 5, H_k, W_k, device=device)
                        for kk in range(positions_k.shape[0]):
                            rr, cc = int(positions_k[kk, 0]), int(positions_k[kk, 1])
                            ak = int(actions_k[kk].item())
                            if 0 <= ak < 5:
                                A_prev[0, ak, rr, cc] = 1.0
                        # Conflict features
                        C_full = compute_conflict_features(
                            positions_k.to(device), actions_k, feat[0].to(device), H_k, W_k,
                        )  # [11, H, W]
                        C_prev = C_full.unsqueeze(0)
                        # Early stop if no conflict detected.
                        if _k + 1 < _kmax:
                            if C_full[8].sum().item() == 0:
                                break
                else:
                    logits, _ = model(feat_dev)
            positions, agent_ids = extract_agent_positions(feat)
            if positions.shape[0] == 0:
                feat, rews, done, _ = env.step(np.zeros(N, dtype=np.int64))
                reached |= (rews >= 9.0)
                if done: break
                continue

            ag_logits = logits[0, :, positions[:, 0], positions[:, 1]].T
            obstacle = feat[0]
            next_pos = positions.unsqueeze(1) + deltas.unsqueeze(0)
            nr = next_pos[:, :, 0]; nc = next_pos[:, :, 1]
            in_bounds = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
            valid = in_bounds.clone()
            if in_bounds.any():
                valid[in_bounds] = obstacle[nr[in_bounds], nc[in_bounds]] == 0
            valid[:, 0] = True
            ag_logits = ag_logits.masked_fill(~valid.to(device), float("-inf"))

            actions_vis = ag_logits.argmax(dim=-1).cpu()
            full = np.zeros(N, dtype=np.int64)
            for k_i, a_i in enumerate((agent_ids - 1).long().clamp(0, N - 1).tolist()):
                full[a_i] = int(actions_vis[k_i].item())
            feat, rews, done, _ = env.step(full)
            reached |= (rews >= 9.0)
            if done: break
        successes.append(reached.sum() / N)

    arr = np.array(successes)
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, se, arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--num_agents", type=int, default=4)
    p.add_argument("--map_size", type=int, default=16)
    p.add_argument("--density", type=float, default=0.3)
    p.add_argument("--max_steps", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    pretrained = os.path.join(_SEAM, "results/checkpoints/railgun_pretrained.pt")
    # Override via env var for the PPO checkpoint:
    ppo_best = os.environ.get("PPO_BEST",
                              os.path.join(_SEAM, "runs/ppo_v18a_moe/best.pt"))

    out_path = os.path.join(_SEAM, "results/pretrained_diag/L_tight_compare.json")
    results = {}
    print(f"Tight eval: {args.episodes} episodes, masked, POGEMA random "
          f"{args.num_agents}ag/{args.map_size}x{args.map_size} (density={args.density})\n")
    for label, path in [("pretrained", pretrained), ("ppo_best", ppo_best)]:
        print(f"  {label} ({path})")
        mean, se, arr = evaluate(path, n_episodes=args.episodes,
                                  num_agents=args.num_agents,
                                  map_size=args.map_size,
                                  density=args.density,
                                  max_steps=args.max_steps,
                                  seed=args.seed)
        ci_lo, ci_hi = mean - 1.96 * se, mean + 1.96 * se
        print(f"    ISR = {mean:.4f}  ± {se:.4f}  (95% CI [{ci_lo:.3f}, {ci_hi:.3f}])\n")
        results[label] = {"isr_mean": mean, "isr_se": se,
                          "isr_ci95": [float(ci_lo), float(ci_hi)],
                          "ckpt": path}

    delta = results["ppo_best"]["isr_mean"] - results["pretrained"]["isr_mean"]
    delta_se = math.sqrt(results["ppo_best"]["isr_se"] ** 2 + results["pretrained"]["isr_se"] ** 2)
    z = delta / delta_se if delta_se > 0 else 0.0
    print(f"  Δ(ppo - pretrained) = {delta:+.4f}  ± {delta_se:.4f}  (z={z:+.2f})")
    results["delta"] = {"value": float(delta), "se": float(delta_se), "z": float(z)}
    results["config"] = {
        "episodes": args.episodes,
        "num_agents": args.num_agents,
        "map_size": args.map_size,
        "density": args.density,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "distribution": "pogema_random",
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
