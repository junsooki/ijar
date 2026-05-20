# IJAR — Iterative Joint-Action Refinement for MAPF

A lightweight neural method for **multi-agent pathfinding (MAPF)** that augments a centralized policy with explicit, iterative conflict-feedback refinement. Built on top of [RAILGUN](https://github.com/airi-institute/rail-gun) as the backbone.

> **Status (May 2026):** On the hard POGEMA dense regime (32 agents, 16×16, density 0.30), n=200 tight eval:
>
> | Method                          | ISR                |
> |---------------------------------|--------------------|
> | RAILGUN baseline                | 0.286              |
> | Pure PIBT                       | 0.461              |
> | **IJAR (17K LACAM samples)**    | **0.582 ± 0.009**  |
>
> Δ vs pure PIBT = +0.121 (z = +23.4). No test-time search; pure NN argmax with K=2 refinement passes.

---

## What it is

A centralized MAPF policy that, instead of producing actions in one forward pass, produces a **draft joint action**, computes a deterministic **conflict feature map** from that draft, and refines its output conditioned on the draft + conflict map. At inference, the loop runs up to K passes with early-stop on no-conflict.

Mathematically:

```
P^r = f_θ(X, A^(r-1), C^(r-1))    with    A^0 = 0, C^0 = 0
```

where `X` is the state, `A^(r-1)` is the previous-pass action proposal (5-channel one-hot at agent cells), `C^(r-1)` is the 11-channel conflict feature map computed from the proposal, and `P^r` is the refined logits.

---

## Architecture

```
              X [6, H, W]
                  ↓
         RAILGUN U-Net body (preserved, ~31M params)
                  ↓ [64, H, W] penultimate
                  ⊕  ← additive combine
                  ↑
   FeedbackEncoder (NEW, 16K params, zero-init final layer)
                  ↑
         (A^(r-1), C^(r-1)) concatenated [16, H, W]
                  ↓
        output_conv (RAILGUN's 1×1)
                  ↓
              P^r [5, H, W]
                  ↓
   argmax (with action mask) → A^r
   compute_conflict_features(A^r) → C^r
                  ↓
    repeat until no conflict or K_max passes
```

**Conflict feature channels (deterministic, no learnable params):**

| Channel | Description |
|---|---|
| 0–4 | Action one-hot at each agent's current cell |
| 5   | Proposed next-cell occupancy count (normalized) |
| 6   | Vertex-conflict indicator |
| 7   | Edge-swap conflict indicator |
| 8   | "Agent is in any conflict" indicator |
| 9   | "Proposed target is an obstacle" indicator |
| 10  | Reserved (entropy / confidence) |

---

## Repo layout

```
ijar/
├── train_refinement.py             # SL training loop (two-pass loss + scheduled sampling)
├── tools/
│   ├── iterative_refinement.py     # IterativeRefinementPolicy + FeedbackEncoder
│   ├── conflict_features.py        # deterministic 11-channel conflict map
│   ├── gen_lacam_hard.py           # LACAM expert data generation
│   ├── diag_pogema_compare.py      # tight n=200 evaluation
│   ├── eval_lacam_matched_budget.py # LaCAM-at-matched-compute baseline
│   ├── graph_construction.py       # agent position extraction
│   └── tight_eval.py
├── envs/
│   └── pogema_railgun_env.py       # POGEMA → RAILGUN feature adapter
├── vessl/                          # GPU run configs (multi-seed + multi-regime)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Conda env
conda create -n ijar python=3.11 -y
conda activate ijar

# 2. PyTorch (Apple Silicon — MPS)
conda install pytorch torchvision -c pytorch -y
#    On Linux + CUDA:
# conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. Other deps
pip install -r requirements.txt

# 4. RAILGUN (clone as sibling repo, then install)
git clone <railgun_repo_url> RAILGUN
cd RAILGUN && pip install -e . && cd ..

# 5. Place the pretrained RAILGUN checkpoint at:
#    results/checkpoints/railgun_pretrained.pt
```

---

## Data generation

The training dataset is built by running LACAM on randomly-sampled hard-regime POGEMA maps and saving each `(state, expert_action)` tuple from solved trajectories.

```bash
# Generate ~5K samples on the baseline hard regime
python tools/gen_lacam_hard.py \
  --num_maps 1000 --seed 42 --time_limit_sec 5 \
  --num_agents 32 --map_size 16 --density 0.30 \
  --out data/lacam_hard.pt

# Append additional samples for scaling experiments
python tools/gen_lacam_hard.py \
  --num_maps 1000 --seed 43 --time_limit_sec 5 \
  --append_existing data/lacam_hard.pt \
  --out data/lacam_hard_combined.pt
```

LACAM solve rate on the hard regime is ~13% within a 5-second budget; each solved episode contributes ~40 (state, action) tuples.

---

## Training

```bash
python train_refinement.py \
  --dataset data/lacam_hard_combined.pt \
  --unet_checkpoint results/checkpoints/railgun_pretrained.pt \
  --epochs 30 --batch_size 16 \
  --val_frac 0.05 --early_stop_patience 100 \
  --seed 42 \
  --run_name refinement_hard32
```

Notable hyperparameters:
- **Two-pass loss:** `L = 0.5·CE(P^1, A*) + 1.0·CE(P^2, A*)`
- **Scheduled sampling for pass-2 input:** 70% model's own first-pass argmax / 20% noised expert (20% of agents corrupted) / 10% clean expert
- **Optimizer:** AdamW, lr=1e-4, batch_size=16

Each epoch on a Mac mini (MPS) with ~17K samples takes ~16 minutes. 30 epochs is ~8 hours.

---

## Evaluation

Tight n=200 evaluation, K=2 refinement passes, action mask applied:

```bash
REFINEMENT_KMAX=2 PPO_BEST=runs/refinement_hard32/final.pt \
  python tools/diag_pogema_compare.py \
  --episodes 200 --num_agents 32 --map_size 16 --density 0.30 \
  --max_steps 128 --seed 42
```

`PPO_BEST` is the env-var override for the checkpoint path. `REFINEMENT_KMAX` controls how many refinement passes per env step (default 3).

---

## GPU / VESSL

For multi-seed and multi-regime sweeps, GPU is ~8× faster than MPS. Configs:

- `vessl/train_refinement.yaml` — single-seed run
- `vessl/multi_seed_sweep.yaml` — 3 seeds in parallel
- `vessl/regime_sweep.yaml` — 5 regimes in parallel

See `vessl/README.md` for prereqs (push repo to GitHub, upload dataset to VESSL volume, etc.).

---

## Action space

RAILGUN action indices (mapped to POGEMA):

| Index | Action | Δ(row, col) |
|---|---|---|
| 0 | Stay | (0, 0) |
| 1 | Right | (0, +1) |
| 2 | Left | (0, -1) |
| 3 | Up | (-1, 0) |
| 4 | Down | (+1, 0) |
