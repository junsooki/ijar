# CRAFT — Cooperative Reward-shaping And Fine-Tuning

CRAFT fine-tunes a pre-trained [RAILGUN](https://github.com/airi-institute/rail-gun) UNet for **multi-agent path finding (MAPF)** using PPO reinforcement learning with heuristic cooperative cost shaping.

The core hypothesis: RAILGUN's frozen UNet fails to cooperate when agents are densely packed. CRAFT diagnoses those failures and corrects them with shaped reward signals — without retraining the backbone.

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
| **Betweenness centrality** | Pre-cached graph centrality score of the cell (bottleneck detection) | `w_bottleneck = 3.0` |
| **Directional conflict** | Whether another agent's heading points at this agent's next cell | `w_conflict = 5.0` |

Cost shaping subtracts a proximity-weighted penalty from the UNet logits before sampling:

```
shaped_logit[a, action] -= alpha * sum_over_neighbors(
    max(0, (radius + 1 - dist) / (radius + 1)) * cost[neighbor]
)
```

This provides dense gradient signal even when agents are not occupying the same cell.

### 4. PPO Training Loop (`train_rl.py`)

```
for iteration in range(max_iterations):
    1. Collect rollout (T steps × N agents)
       - Forward UNet → shape with phi costs → sample actions
       - Store (obs, action, log_prob, reward, value, done)

    2. Compute GAE (γ=0.99, λ=0.95)
       - Normalize advantages

    3. PPO update (multiple epochs over minibatches)
       - Clipped surrogate policy loss (ε=0.2)
       - Value loss (MSE)
       - Entropy bonus (coef=0.01)
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
craft/
├── train_rl.py                  # Main PPO training script
├── configs/
│   └── rl_ppo.yaml              # All hyperparameters
├── envs/
│   └── pogema_railgun_env.py    # POGEMA → RAILGUN feature adapter
├── tools/
│   └── heuristic_cost.py        # Cooperative cost shaping
├── docs/
│   └── explain_phi_gnn.py       # Educational PDF generator
├── diagnostic.ipynb             # Colab diagnostic notebook
├── diagnostic_local.ipynb       # Local diagnostic notebook
├── colab_diagnostic_plan.md     # Phase 1 analysis plan
├── requirements.txt
└── results/                     # Generated reports
```

---

## Diagnostic Analysis

Before algorithm work, CRAFT runs a structured diagnostic to **prove** the cooperation failure hypothesis:

1. **Data collection** — generate maps, run RAILGUN inference, log per-cell records (action distribution, disagreement vs LaCAM expert, local density, betweenness)
2. **Correlation analysis** — does disagreement rate increase with agent density?
3. **Criticality metrics** — betweenness centrality, path intersections, revisit count
4. **AUC test** — can density alone predict disagreement?
5. **Spatial heatmaps** — do failures cluster at bottlenecks?

See `diagnostic.ipynb` / `colab_diagnostic_plan.md` for details.

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# RAILGUN must be installed separately (provides UNet, cost shaping, feature builder)
# See: https://github.com/airi-institute/rail-gun
```

## Training

```bash
python train_rl.py --config configs/rl_ppo.yaml --unet_checkpoint path/to/railgun.pt
```

Key config options (`configs/rl_ppo.yaml`):

```yaml
unet_checkpoint: null          # Path to pretrained RAILGUN UNet
num_agents: 4                  # Starting curriculum stage
learning_rate: 3.0e-5
phi_w_density: 2.0             # Density cost weight
phi_w_bottleneck: 3.0          # Bottleneck cost weight
phi_w_conflict: 5.0            # Directional conflict cost weight
```

TensorBoard logs are written to `runs/` and checkpoints to `checkpoints/`.

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
