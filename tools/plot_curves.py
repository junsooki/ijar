"""Generate training curves from runs/<run>/audit.jsonl + stdout caches.

Usage:
    python3 tools/plot_curves.py --runs v15_seed2 v16_nophi --out seam_curves.png

Reads each run's audit.jsonl for entropy, mean_reward, isr (if logged), and
falls back to /tmp/seam_<run>_output.txt for ISR/reward when audit doesn't
have them (older runs).
"""

import argparse
import json
import os
import re
import sys

import matplotlib.pyplot as plt

ISR_RE = re.compile(r"ISR\s*=\s*([\d.]+)")
REW_RE = re.compile(r"\[\s*\d+/\d+\]\s+rew=([-\d.]+)")
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
          "tab:brown", "tab:gray", "tab:pink", "tab:olive", "tab:cyan"]


def parse_run(run_name, runs_dir, stdout_dir):
    """Return dict with arrays: iters, isrs, rews, ents, isr_iters."""
    audit_path = os.path.join(runs_dir, run_name, "audit.jsonl")
    audit = []
    if os.path.isfile(audit_path):
        with open(audit_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    audit.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # mean_reward and entropy from audit (if present), else from stdout
    iters, rews, ents = [], [], []
    isr_iters_audit, isrs_audit = [], []
    for d in audit:
        it = d.get("iteration")
        if it is None:
            continue
        if "ppo_entropy" in d:
            iters.append(it)
            ents.append(d["ppo_entropy"])
            rews.append(d.get("mean_reward", None))
        if "isr" in d:
            isr_iters_audit.append(it)
            isrs_audit.append(d["isr"])

    # Fall back to stdout for missing reward / isr
    stdout_path = os.path.join(stdout_dir, f"seam_{run_name}_output.txt")
    isr_iters, isrs = isr_iters_audit, isrs_audit
    rews_from_stdout = []
    if os.path.isfile(stdout_path):
        with open(stdout_path) as f:
            iter_count = 0
            for line in f:
                m = REW_RE.search(line)
                if m:
                    iter_count += 1
                    rews_from_stdout.append(float(m.group(1)))
                if not isr_iters_audit:
                    m2 = ISR_RE.search(line)
                    if m2 and "→" in line:
                        # Each ISR line corresponds to current iter_count
                        isrs.append(float(m2.group(1)))
                        isr_iters.append(iter_count)

    # If audit didn't give us rewards, replace with stdout
    if rews and any(r is None for r in rews) and rews_from_stdout:
        rews = rews_from_stdout[:len(iters)]

    return {
        "name": run_name,
        "iters": iters,
        "rews": [r if r is not None else float("nan") for r in rews],
        "ents": ents,
        "isr_iters": isr_iters,
        "isrs": isrs,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True,
                    help="run directory names under runs/")
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--stdout_dir", default="/tmp",
                    help="where seam_<run>_output.txt lives")
    p.add_argument("--out", default="seam_curves.png")
    p.add_argument("--title", default="SEAM training curves")
    args = p.parse_args()

    parsed = [parse_run(r, args.runs_dir, args.stdout_dir) for r in args.runs]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for i, d in enumerate(parsed):
        c = COLORS[i % len(COLORS)]
        axes[0].plot(d["isr_iters"], d["isrs"], marker="o", linewidth=2,
                     markersize=5, color=c, label=d["name"])
        axes[1].plot(d["iters"], d["rews"], color=c, alpha=0.7, label=d["name"])
        axes[2].plot(d["iters"], d["ents"], color=c, alpha=0.7, label=d["name"])

    axes[0].set_ylabel("ISR")
    axes[0].set_title(args.title)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=9)

    axes[1].set_ylabel("mean reward / step")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("policy entropy")
    axes[2].set_xlabel("training iteration")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")
    for d in parsed:
        n_isr = len(d["isrs"])
        n_rew = len([r for r in d["rews"] if r == r])  # not nan
        print(f"  {d['name']}: {n_isr} ISR points, {n_rew} reward points,"
              f" {len(d['ents'])} entropy points")


if __name__ == "__main__":
    main()
