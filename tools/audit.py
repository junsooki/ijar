"""Training audit logger for SEAM.

Captures per-iteration stats useful for tuning the phi equations:
  - Phi signal breakdown (density / bottleneck / conflict contributions)
  - PPO health (clip fraction, KL estimate, grad norm)
  - Action histogram and freeze rate
"""

import json
import os
from collections import defaultdict

import numpy as np
import torch

ACTION_NAMES = ["stay", "right", "left", "up", "down"]


class AuditLogger:
    def __init__(self, log_dir: str, writer, print_interval: int = 10):
        self.writer = writer
        self.print_interval = print_interval
        self._jsonl_path = os.path.join(log_dir, "audit.jsonl")
        self._fh = open(self._jsonl_path, "a")

    # ── Public API ────────────────────────────────────────────────────────────

    def record(self, iteration: int, rollout_stats: dict, ppo_metrics: dict,
               mean_reward: float | None = None, isr: float | None = None):
        """Write to TensorBoard and JSONL; print summary every print_interval iters.

        mean_reward and isr are optional curve-tracking fields; if provided
        they're written to the JSONL line so plot_curves.py can read them
        without parsing stdout.
        """
        self._log_tb(iteration, rollout_stats, ppo_metrics)
        extra = {}
        if mean_reward is not None:
            extra["mean_reward"] = float(mean_reward)
        if isr is not None:
            extra["isr"] = float(isr)
        self._log_jsonl(iteration, rollout_stats, ppo_metrics, extra=extra)
        if iteration % self.print_interval == 0:
            self.print_summary(iteration, rollout_stats, ppo_metrics)

    def close(self):
        self._fh.close()

    # ── Console summary ───────────────────────────────────────────────────────

    def print_summary(self, iteration: int, rollout_stats: dict, ppo_metrics: dict):
        phi  = rollout_stats.get("phi", {})
        act  = rollout_stats.get("actions", {})
        ppo  = ppo_metrics

        # Phi signal breakdown as % of total phi mass
        d = phi.get("mean_density", 0.0)
        b = phi.get("mean_bottleneck", 0.0)
        c = phi.get("mean_conflict", 0.0)
        total = d + b + c + 1e-9
        dpct, bpct, cpct = 100*d/total, 100*b/total, 100*c/total

        # Action distribution
        hist = act.get("histogram", [0.2]*5)
        act_str = "  ".join(f"{n}={100*p:.0f}%" for n, p in zip(ACTION_NAMES, hist))

        print(
            f"  PHI  mean={phi.get('mean_phi', 0):.2f}  max={phi.get('max_phi', 0):.2f}"
            f"  nonzero={100*phi.get('frac_nonzero', 0):.0f}%"
            f"  [density={dpct:.0f}% bn={bpct:.0f}% conf={cpct:.0f}%]"
        )
        print(
            f"  PPO  clip={100*ppo.get('clip_frac', 0):.1f}%"
            f"  kl≈{ppo.get('approx_kl', 0):.4f}"
            f"  grad_norm={ppo.get('grad_norm', 0):.3f}"
        )
        print(f"  ACT  {act_str}  freeze={100*act.get('freeze_rate', 0):.1f}%")

    # ── TensorBoard ───────────────────────────────────────────────────────────

    def _log_tb(self, iteration: int, rollout_stats: dict, ppo_metrics: dict):
        w = self.writer
        phi = rollout_stats.get("phi", {})
        act = rollout_stats.get("actions", {})

        w.add_scalar("Phi/mean_phi",        phi.get("mean_phi", 0),        iteration)
        w.add_scalar("Phi/max_phi",         phi.get("max_phi", 0),         iteration)
        w.add_scalar("Phi/frac_nonzero",    phi.get("frac_nonzero", 0),    iteration)
        w.add_scalar("Phi/mean_density",    phi.get("mean_density", 0),    iteration)
        w.add_scalar("Phi/mean_bottleneck", phi.get("mean_bottleneck", 0), iteration)
        w.add_scalar("Phi/mean_conflict",   phi.get("mean_conflict", 0),   iteration)

        w.add_scalar("PPO/clip_frac",   ppo_metrics.get("clip_frac", 0),   iteration)
        w.add_scalar("PPO/approx_kl",   ppo_metrics.get("approx_kl", 0),   iteration)
        w.add_scalar("PPO/grad_norm",   ppo_metrics.get("grad_norm", 0),    iteration)

        hist = act.get("histogram", [])
        for name, val in zip(ACTION_NAMES, hist):
            w.add_scalar(f"Actions/{name}", val, iteration)
        w.add_scalar("Actions/freeze_rate", act.get("freeze_rate", 0), iteration)

    # ── JSONL ─────────────────────────────────────────────────────────────────

    def _log_jsonl(self, iteration: int, rollout_stats: dict, ppo_metrics: dict,
                   extra: dict | None = None):
        record = {"iteration": iteration}
        record.update({f"phi_{k}": v for k, v in rollout_stats.get("phi", {}).items()})
        record.update({f"act_{k}": v for k, v in rollout_stats.get("actions", {}).items()})
        record.update({f"ppo_{k}": v for k, v in ppo_metrics.items()})
        if extra:
            record.update(extra)
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()


# ── Stats collectors (called from train_rl.py) ─────────────────────────────

def collect_phi_stats(
    phi_costs_list: list,        # list of [N] tensors (one per rollout step)
    components_list: list,       # list of component dicts
) -> dict:
    """Aggregate phi stats across a rollout."""
    if not phi_costs_list:
        return {}

    all_phi = torch.cat([p.cpu() for p in phi_costs_list])   # [T*N]
    all_den = torch.cat([c["density"].cpu()    for c in components_list])
    all_bn  = torch.cat([c["bottleneck"].cpu() for c in components_list])
    all_con = torch.cat([c["conflict"].cpu()   for c in components_list])

    return {
        "mean_phi":       all_phi.mean().item(),
        "max_phi":        all_phi.max().item(),
        "frac_nonzero":   (all_phi > 0).float().mean().item(),
        "mean_density":    all_den.mean().item(),
        "mean_bottleneck": all_bn.mean().item(),
        "mean_conflict":   all_con.mean().item(),
    }


def collect_action_stats(
    actions_list: list,          # list of [N] int64 tensors (one per rollout step)
    positions_list: list,        # list of [N, 2] tensors
) -> dict:
    """Action histogram and freeze rate across a rollout."""
    if not actions_list:
        return {}

    all_actions = torch.cat([a.cpu() for a in actions_list])   # [T*N]
    counts = torch.bincount(all_actions, minlength=5).float()
    hist = (counts / counts.sum()).tolist()

    # Freeze rate: fraction of (step, agent) pairs where agent didn't move
    # Detected by comparing consecutive positions
    freeze_count = 0
    total = 0
    for t in range(1, len(positions_list)):
        p_cur, p_prev = positions_list[t], positions_list[t - 1]
        if p_cur.shape[0] != p_prev.shape[0]:
            continue  # skip when agent count changes (someone reached goal)
        moved = (p_cur - p_prev).abs().sum(dim=-1) == 0
        freeze_count += moved.sum().item()
        total += moved.numel()

    return {
        "histogram":   hist,
        "freeze_rate": freeze_count / max(total, 1),
    }
