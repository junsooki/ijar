"""Generate more LACAM expert trajectories on hard regime (32ag/16x16/d=0.30).

Adapted from archive/2026-05-mppi-investigation/tools/recipe_c_lacam_imitation.py
to ONLY do data collection (no training). Output is appended/saved to a new
dataset file for refinement training.

Usage:
    PYTHONNOUSERSITE=1 python tools/gen_lacam_hard.py --num_maps 300 --seed 1000 \
        --out data/lacam_hard_extra.pt
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SEAM = os.path.dirname(_HERE)
sys.path.insert(0, _SEAM)
sys.path.append(os.path.join(_SEAM, "RAILGUN"))
sys.path.append(os.path.join(_SEAM, "RAILGUN", "tools", "extensions"))

from envs.pogema_railgun_env import POGEMARailgunEnv, NOT_FOUND_DIST, bfs_distance_from_goal
import lacam_online_native as lacam

NOT_FOUND = 2048


def dump_map_file(obstacle: np.ndarray, path: str):
    H, W = obstacle.shape
    with open(path, "w") as f:
        f.write("type octile\n")
        f.write(f"height {H}\n")
        f.write(f"width {W}\n")
        f.write("map\n")
        for r in range(H):
            row = ""
            for c in range(W):
                row += "@" if obstacle[r, c] else "."
            f.write(row + "\n")


def build_feature_offline(obstacle: np.ndarray,
                          positions: np.ndarray,
                          goals: np.ndarray) -> torch.Tensor:
    """Reconstruct RAILGUN's [6, H, W] feature for feature_type='none'."""
    H, W = obstacle.shape
    feat = torch.zeros(6, H, W, dtype=torch.float32)
    feat[0] = torch.from_numpy(obstacle.astype(np.float32))
    N = positions.shape[0]
    for i in range(N):
        r, c = int(positions[i, 0]), int(positions[i, 1])
        if 0 <= r < H and 0 <= c < W:
            feat[1, r, c] = float(i + 1)
        gr, gc = int(goals[i, 0]), int(goals[i, 1])
        if 0 <= gr < H and 0 <= gc < W:
            feat[2, gr, gc] = float(i + 1)
    # Channels 4, 5: signed displacement (per RAILGUN convention for feature_type='none')
    for i in range(N):
        r, c = int(positions[i, 0]), int(positions[i, 1])
        gr, gc = int(goals[i, 0]), int(goals[i, 1])
        if 0 <= r < H and 0 <= c < W:
            feat[4, r, c] = float(gr - r)
            feat[5, r, c] = float(gc - c)
    return feat


def collect(num_maps: int, num_agents: int, map_size: int, density: float,
            time_limit_sec: int, seed: int):
    rng = np.random.RandomState(seed)
    env = POGEMARailgunEnv(
        map_source=None, num_agents=num_agents, max_steps=128,
        density=density, size=map_size, feature_type="none",
    )
    samples = []
    n_solved = 0
    n_failed = 0
    for k in range(num_maps):
        env.reset(seed=int(rng.randint(0, 2**31)))
        obstacle = env.obstacle.copy()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".map", delete=False) as f:
            map_path = f.name
        dump_map_file(obstacle, map_path)
        try:
            sol = lacam.generate_lacam_solution_cpp(
                map_file=map_path,
                agent_num=num_agents,
                seed=int(rng.randint(0, 2**31)),
                time_limit_sec=time_limit_sec,
            )
        except Exception:
            n_failed += 1
            try: os.unlink(map_path)
            except OSError: pass
            continue
        finally:
            try: os.unlink(map_path)
            except OSError: pass

        positions = sol["positions"]
        actions   = sol["actions"]
        goals     = sol["goals"]
        T = positions.shape[0]
        n_solved += 1
        for t in range(T - 1):
            feat = build_feature_offline(obstacle, positions[t], goals)
            samples.append((feat, torch.from_numpy(actions[t].astype(np.int64))))
        if (k + 1) % 20 == 0:
            print(f"  [{k+1}/{num_maps}] solved={n_solved} failed={n_failed} samples={len(samples)}", flush=True)
    print(f"Done: solved={n_solved} failed={n_failed} → {len(samples)} samples")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_maps", type=int, default=300)
    parser.add_argument("--num_agents", type=int, default=32)
    parser.add_argument("--map_size", type=int, default=16)
    parser.add_argument("--density", type=float, default=0.30)
    parser.add_argument("--time_limit_sec", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--append_existing", type=str, default=None,
                        help="if given, concatenate the existing dataset path with newly collected samples")
    args = parser.parse_args()

    samples = collect(
        num_maps=args.num_maps, num_agents=args.num_agents,
        map_size=args.map_size, density=args.density,
        time_limit_sec=args.time_limit_sec, seed=args.seed,
    )
    if args.append_existing:
        print(f"Loading existing {args.append_existing} ...")
        prev = torch.load(args.append_existing, map_location="cpu", weights_only=False)
        print(f"  existing: {len(prev)} samples")
        samples = prev + samples
        print(f"  combined: {len(samples)} samples")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(samples, args.out)
    print(f"Saved {len(samples)} samples to {args.out}")


if __name__ == "__main__":
    main()
