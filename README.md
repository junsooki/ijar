# SEAM — Social-cost Enhancement for Agent Movement

SEAM fine-tunes a pre-trained [RAILGUN](https://github.com/airi-institute/rail-gun) UNet for **multi-agent path finding (MAPF)** using PPO reinforcement learning with heuristic cooperative cost shaping.

The core hypothesis: RAILGUN's frozen UNet fails to cooperate when agents are densely packed. SEAM diagnoses those failures and corrects them with shaped reward signals — without retraining the backbone.

---

## How It Works

```
Pretrained UNet (frozen, 7.8M params)
       ↓
  Raw logits [5, H, W]   ←── 6-channel feature map from environment
       ↓
Heuristic Phi Adapter    ←── density + bottleneck + conflict costs
       ↓
Shaped logits → action sampling
       ↓
PPO update on Value Head only (+ fine-tune policy logits)
```

### 1. Observation Representation

Each timestep the environment produces a `[6, H, W]` tensor:

| Channel | Content |
|---------|---------|
| 0 | Obstacle map (1 = wall) |
| 1 | Agent positions (value = agent ID) |
| 2 | Goal positions (value = agent ID) |
| 3 | BFS distance-to-goal, normalized by H+W |
| 4 | Distance gradient X (row direction) |
| 5 | Distance gradient Y (column direction) |

### 2. Model Architecture

- **UNet backbone** — frozen pre-trained RAILGUN weights; outputs `[5, H, W]` action logits over the entire grid
- **Value Head** — `GlobalAvgPool(logits) → Linear(5, 64) → ReLU → Linear(64, 1)`; trained from scratch

### 3. Heuristic Phi — Cooperative Cost (`tools/heuristic_cost.py`)

Three signals are summed per agent to produce a cooperation cost:

| Signal | Description | Default weight |
|--------|-------------|----------------|
| **Local density** | Number of other agents within L-inf radius 3 | `w_density = 2.0` |
| **Betweenness centrality** | Normalized graph centrality of the cell — higher at bottlenecks | `w_bottleneck = 3.0` |
| **Directional conflict** | Soft dot-product alignment of j's heading toward i, weighted by proximity | `w_conflict = 5.0` |

Cost shaping subtracts a proximity-weighted, logit-scale-normalized penalty from UNet logits before sampling:

```
effective_alpha = phi_alpha / std(UNet logits at agent positions)

shaped_logit[i, a] -= effective_alpha * Σⱼ  max(0, (R+1 - dist) / (R+1)) * φⱼ
```

Both the victim (agent being headed toward) and the aggressor (agent doing the heading) receive the conflict penalty.

### 4. PPO Training Loop (`train_rl.py`)

```
for iteration in range(max_iterations):
    1. Collect rollout (T steps × N agents)
       - Forward UNet → normalize alpha by logit scale → shape with phi costs
       - Store (obs, action, shaped_logits, shaping_delta, reward, value, done)

    2. Compute GAE (γ=0.99, λ=0.95)
       - Normalize advantages

    3. PPO update (multiple epochs over minibatches)
       - new_log_prob: raw UNet logits + stored shaping_delta (same distribution as rollout)
       - old_log_prob: recomputed from stored shaped_agent_logits
       - Clipped surrogate loss (ε=0.2), value loss (MSE), entropy bonus (0.01)
       - Gradient clip (max_norm=0.5)

    4. Evaluate ISR every eval_interval
       - ISR = agents reaching goal / total agents
       - Save checkpoint on improvement
       - Advance curriculum if threshold met
```

### 5. Curriculum Learning

Three stages, each gated by Individual Success Rate (ISR):

| Stage | Agents | Map size | Max steps | ISR to advance |
|-------|--------|----------|-----------|----------------|
| 1 | 4 | 16×16 | 128 | 0.50 |
| 2 | 8 | 16×16 | 192 | 0.50 |
| 3 | 16 | 32×32 | 256 | — |

---

## Project Structure

```
seam/
├── train_rl.py                  # Main PPO training script
├── configs/
│   └── rl_ppo.yaml              # All hyperparameters
├── envs/
│   └── pogema_railgun_env.py    # POGEMA → RAILGUN feature adapter
├── tools/
│   ├── heuristic_cost.py        # Cooperative cost shaping (phi)
│   └── audit.py                 # Training audit logger
├── docs/
│   ├── seam_explained.py        # ELI4 PDF generator
│   └── seam_explained.pdf       # Pre-built explainer PDF
├── diagnostic.ipynb             # Colab diagnostic notebook
├── diagnostic_local.ipynb       # Local diagnostic notebook
├── requirements.txt
└── results/                     # Generated reports
```

---

## Diagnostic Analysis

SEAM includes a structured diagnostic to confirm the cooperation failure hypothesis:

1. **Data collection** — generate maps, run RAILGUN inference, log per-cell records (action distribution, disagreement vs LaCAM expert, local density, betweenness)
2. **Correlation analysis** — does disagreement rate increase with agent density?
3. **Criticality metrics** — betweenness centrality, path intersections, revisit count
4. **AUC test** — can density alone predict disagreement?
5. **Spatial heatmaps** — do failures cluster at bottlenecks?

Requires RAILGUN installed + a pretrained checkpoint. See `diagnostic.ipynb` for details.

---

## Running on Mac (Apple Silicon)

**Short answer: yes, test runs work fine on an M4 Mac mini.**

The training script auto-detects MPS (Metal) → falls back to CPU. With 32 GB unified memory and an M4 chip:

| Setting | Mac mini M4 (32 GB) | Notes |
|---------|-------------------|-------|
| Stage 1 — 4 agents, 16×16 | ✅ runs well | ~2–5 it/s on MPS |
| Stage 2 — 8 agents, 16×16 | ✅ runs fine | slightly slower |
| Stage 3 — 16 agents, 32×32 | ⚠️ slow but possible | expect 0.5–1 it/s |
| Full curriculum to convergence | ❌ not practical | needs GPU cluster |

Good for: smoke-testing the pipeline, debugging env logic, verifying reward signal, short ablations. Not good for: training to full ISR convergence (needs a CUDA GPU for hours/days).

---

## Setup

### 1. Create Conda Environment

```bash
conda create -n seam python=3.11 -y
conda activate seam
```

### 2. Install PyTorch (Apple Silicon / MPS)

```bash
# Apple Silicon Mac — installs PyTorch with MPS support
conda install pytorch torchvision -c pytorch -y
```

> On Linux with CUDA, replace with:
> `conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y`

### 3. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install RAILGUN

RAILGUN is not on PyPI — clone and install it manually:

```bash
git clone <railgun_repo_url> RAILGUN
cd RAILGUN && pip install -e . && cd ..
```

### 5. Verify Setup

```bash
python - <<'EOF'
import torch, pogema, networkx, yaml
print("torch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("pogema:", pogema.__version__)
print("All good!")
EOF
```

---

## Training

### Quick test run on Mac (no RAILGUN checkpoint needed)

```bash
python train_rl.py --config configs/rl_ppo.yaml
```

This uses a randomly initialized UNet — enough to verify the pipeline runs end-to-end. You should see rollout collection, PPO updates, and ISR logged to console.

### Full training with pretrained UNet

```bash
python train_rl.py --config configs/rl_ppo.yaml --unet_checkpoint path/to/railgun.pt
```

Key config options (`configs/rl_ppo.yaml`):

```yaml
unet_checkpoint: null          # Path to pretrained RAILGUN UNet (null = random init)
num_agents: 4                  # Starting curriculum stage
learning_rate: 3.0e-5
phi_w_density: 2.0             # Density cost weight
phi_w_bottleneck: 3.0          # Bottleneck cost weight
phi_w_conflict: 5.0            # Directional conflict cost weight
```

TensorBoard logs are written to `runs/` and checkpoints to `checkpoints/`.

```bash
tensorboard --logdir runs/
```

---

## Action Space

RAILGUN canonical indices (mapped to POGEMA):

| Index | Action | Delta |
|-------|--------|-------|
| 0 | Stay | [0, 0] |
| 1 | Right | [0, +1] |
| 2 | Left | [0, -1] |
| 3 | Up | [-1, 0] |
| 4 | Down | [+1, 0] |
