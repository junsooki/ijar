# VESSL run scripts for SEAM refinement training

These YAMLs run on Anthropic-internal or commercial VESSL clusters.

## Files

| File | Purpose | Time on L4 GPU |
|---|---|---|
| `train_refinement.yaml` | Single-run training + n=200 eval | ~1.5 hr |
| `multi_seed_sweep.yaml` | 3-seed parallel sweep for variance bars | ~1.5 hr (parallel) |
| `regime_sweep.yaml` | 5-regime parallel sweep for generalization | ~1.5 hr (parallel) |

## Prereqs before running on VESSL

1. **Upload datasets** to a VESSL volume named `seam-lacam-data` (or adjust mount path):
   - `/data/lacam_hard_combined_v4.pt` (25K samples, baseline regime)
   - `/data/regimes/lacam_16ag_16x16_d030.pt` (other regimes — gen with `tools/gen_lacam_hard.py`)
   - `/data/regimes/lacam_48ag_16x16_d030.pt`
   - `/data/regimes/lacam_32ag_d040.pt`
   - `/data/regimes/lacam_32ag_20x20_d030.pt`

2. **Push repo to GitHub** at `github.com/junsooki/seam` (or update `import.git.url`).

3. **Upload RAILGUN checkpoint** to the repo at `results/checkpoints/railgun_pretrained.pt`.

## Local-vs-GPU expected speedup

| Workload | Mac mini (MPS) | L4 GPU (estimate) |
|---|---|---|
| 1 epoch on 17K samples | ~16 min | ~2 min |
| 30 epochs total | ~8 hr | ~1 hr |
| n=200 tight eval | ~15 min (CPU) | ~15 min (unchanged; CPU-bound) |

So ~8× speedup for training. The eval is CPU-bound and unaffected.

## Notes

- We use plain PyTorch (no PyTorch Geometric) — install is light.
- LACAM data generation is included as an optional fallback in `train_refinement.yaml`. Recommended path: pre-generate on Mac and upload.
- Output ckpts and eval logs land in `/output` per VESSL convention.
