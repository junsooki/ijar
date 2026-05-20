"""SL training for iterative joint-action refinement (task #48).

Two-pass training with scheduled sampling:
  - First pass: zero feedback, produce draft action proposal
  - Second pass: condition on action+conflict features derived from a sampled
    proposal (70% model's own first-pass argmax, 20% noised expert, 10% expert)
  - Loss: α·CE(P1, A*) + β·CE(P2, A*)  with α=0.5, β=1.0

Dataset: list of (feature [6, H, W], expert_actions [N]) tuples, loaded from
archived LACAM-on-hard-regime data.

Usage:
    PYTHONNOUSERSITE=1 python train_refinement.py
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
sys.path.append(os.path.join(_HERE, "RAILGUN"))

from models.unet import UNet
from tools.conflict_features import compute_conflict_features, ACTION_DELTAS
from tools.iterative_refinement import IterativeRefinementPolicy
from tools.graph_construction import extract_agent_positions


def load_dataset(path: str) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    print(f"Loading dataset from {path} ...")
    data = torch.load(path, map_location="cpu", weights_only=False)
    print(f"  {len(data)} samples")
    return data


def build_action_map_and_conflicts(
    feat_b: torch.Tensor,        # [6, H, W]
    actions_per_slot: torch.Tensor,  # [N_max] long — actions indexed by agent slot
    H: int, W: int,
    device: torch.device,
):
    """Build a 5-channel action map and 11-channel conflict map from a per-agent
    action vector. Agent positions are extracted from feat channel 1."""
    positions, agent_ids = extract_agent_positions(feat_b.cpu())
    positions = positions.to(device)
    if positions.shape[0] == 0:
        action_map = torch.zeros(5, H, W, device=device)
        conflict_map = torch.zeros(11, H, W, device=device)
        return action_map, conflict_map

    # Map per-slot actions to per-visible-agent actions.
    vis_slots = (agent_ids - 1).long().clamp(0, actions_per_slot.numel() - 1)
    actions_vis = actions_per_slot.to(device)[vis_slots]                       # [K]

    # Action one-hot at agent's current cell.
    action_map = torch.zeros(5, H, W, device=device)
    for k in range(positions.shape[0]):
        r_ = int(positions[k, 0]); c_ = int(positions[k, 1])
        a_ = int(actions_vis[k])
        if 0 <= a_ < 5:
            action_map[a_, r_, c_] = 1.0

    obstacle = feat_b[0].to(device)
    conflict_map = compute_conflict_features(positions, actions_vis, obstacle, H, W)
    return action_map, conflict_map


def actions_from_logits_argmax(
    logits: torch.Tensor,        # [B, 5, H, W]
    feats: torch.Tensor,         # [B, 6, H, W]
    H: int, W: int,
) -> torch.Tensor:
    """Read per-cell logits at each visible agent's position, take argmax.
    Returns [B, N_max] action tensor (zeros for invisible slots)."""
    B = logits.shape[0]
    N_max = 32  # hard regime
    out = torch.zeros(B, N_max, dtype=torch.long, device=logits.device)
    for b in range(B):
        ch1 = feats[b, 1].cpu()
        nz = (ch1 > 0).nonzero(as_tuple=False)
        for k in range(nz.shape[0]):
            r_, c_ = int(nz[k, 0]), int(nz[k, 1])
            aid = int(ch1[r_, c_].item()) - 1
            if 0 <= aid < N_max:
                out[b, aid] = int(torch.argmax(logits[b, :, r_, c_]).item())
    return out


def per_agent_logits_at_positions(
    logits: torch.Tensor,        # [B, 5, H, W]
    feats: torch.Tensor,         # [B, 6, H, W]
    expert_actions: torch.Tensor,  # [B, N_max] long
):
    """Gather logits at each visible agent's cell and pair with expert action.
    Returns (per_agent_logits [M, 5], expert_target [M], valid_mask [M] bool)
    where M = total visible agents in batch."""
    B = logits.shape[0]
    per_agent_logits_list = []
    expert_target_list = []
    for b in range(B):
        ch1 = feats[b, 1].cpu()
        nz = (ch1 > 0).nonzero(as_tuple=False)
        for k in range(nz.shape[0]):
            r_, c_ = int(nz[k, 0]), int(nz[k, 1])
            aid = int(ch1[r_, c_].item()) - 1
            if 0 <= aid < expert_actions.shape[1]:
                per_agent_logits_list.append(logits[b, :, r_, c_])
                expert_target_list.append(expert_actions[b, aid].item())
    if not per_agent_logits_list:
        return None, None
    per_agent = torch.stack(per_agent_logits_list, dim=0)
    target = torch.tensor(expert_target_list, dtype=torch.long, device=logits.device)
    return per_agent, target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="data/lacam_hard.pt",
                        help="Path to LACAM expert dataset (.pt file with list of (state, expert_actions) tuples)")
    parser.add_argument("--unet_checkpoint", type=str,
                        default="results/checkpoints/railgun_pretrained.pt",
                        help="Path to pretrained RAILGUN UNet checkpoint")
    parser.add_argument("--run_name", type=str, default="refinement_hard32")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--val_frac", type=float, default=0.1, help="fraction of dataset held out for validation")
    parser.add_argument("--early_stop_patience", type=int, default=8, help="stop if val acc2 hasn't improved for this many epochs")
    parser.add_argument("--seed", type=int, default=42, help="random seed for model init + data shuffling")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.5, help="first pass loss weight")
    parser.add_argument("--beta", type=float, default=1.0, help="second pass loss weight")
    parser.add_argument("--p_model", type=float, default=0.70, help="scheduled sampling: use model's first-pass argmax")
    parser.add_argument("--p_noised", type=float, default=0.20, help="scheduled sampling: use noised expert")
    parser.add_argument("--p_expert", type=float, default=0.10, help="scheduled sampling: use clean expert")
    parser.add_argument("--noise_p", type=float, default=0.20, help="probability of corrupting each agent's action in noised mode")
    parser.add_argument("--save_dir", type=str, default="runs")
    parser.add_argument("--log_interval", type=int, default=10)
    args = parser.parse_args()

    # Set seeds for reproducibility.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}  seed: {args.seed}")

    # ── data ─────────────────────────────────────────────────────────────────
    data_all = load_dataset(args.dataset)
    # Train / val split (deterministic by seeding numpy here).
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(data_all))
    n_val = max(1, int(len(data_all) * args.val_frac))
    val_idx = set(perm[:n_val].tolist())
    data = [data_all[i] for i in range(len(data_all)) if i not in val_idx]
    data_val = [data_all[i] for i in range(len(data_all)) if i in val_idx]
    print(f"  train: {len(data)}  val: {len(data_val)}")
    # Determine N_max from first sample
    N_max = int(data[0][1].numel())
    print(f"  N_max (agents per sample): {N_max}")
    H, W = data[0][0].shape[-2:]
    print(f"  H, W: {H}, {W}")

    # ── model ────────────────────────────────────────────────────────────────
    unet = UNet(n_channels=6, n_classes=5, first_layer_channels=64,
                bilinear=False, blocks_per_stage=0).to(device)
    if os.path.isfile(args.unet_checkpoint):
        sd = torch.load(args.unet_checkpoint, map_location=device, weights_only=True)
        unet.load_state_dict(sd if "unet" not in sd else sd["unet"], strict=False)
        print(f"Loaded UNet from {args.unet_checkpoint}")
    policy = IterativeRefinementPolicy(unet).to(device)
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in policy.parameters())
    print(f"Policy params: trainable={n_train:,} / total={n_total:,}")

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=args.lr,
    )

    # ── logging ──────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, args.run_name)
    os.makedirs(save_dir, exist_ok=True)

    # ── training loop ────────────────────────────────────────────────────────
    n_samples = len(data)
    n_batches = (n_samples + args.batch_size - 1) // args.batch_size
    print(f"Starting training: {args.epochs} epochs × {n_batches} batches/epoch = {args.epochs * n_batches} updates")

    best_val_acc2 = -1.0
    epochs_since_best = 0
    global_step = 0
    for epoch in range(args.epochs):
        ep_t0 = time.time()
        indices = np.random.permutation(n_samples)
        epoch_loss = 0.0
        epoch_loss1 = 0.0
        epoch_loss2 = 0.0
        epoch_acc1 = 0.0
        epoch_acc2 = 0.0
        n_acc = 0

        for batch_i in range(n_batches):
            batch_idx = indices[batch_i * args.batch_size : (batch_i + 1) * args.batch_size]
            if len(batch_idx) == 0:
                continue
            feats_list  = [data[i][0] for i in batch_idx]
            expert_list = [data[i][1] for i in batch_idx]
            # Pad expert to N_max in case of variable agent counts (here always 32).
            feats = torch.stack(feats_list, dim=0).to(device)              # [B, 6, H, W]
            expert = torch.stack(expert_list, dim=0).long().to(device)     # [B, N_max]
            B = feats.shape[0]

            # ── pass 1 ─────────────────────────────────────────────────────
            policy.train()
            logits1, _ = policy(feats)                                     # [B, 5, H, W]

            # Pass-1 loss on expert.
            pa1, t1 = per_agent_logits_at_positions(logits1, feats, expert)
            if pa1 is None:
                continue
            loss1 = F.cross_entropy(pa1, t1)

            # ── scheduled sampling: pick proposal for pass 2 ───────────────
            with torch.no_grad():
                # Mode per-batch (cheaper than per-sample).
                u = np.random.random()
                if u < args.p_model:
                    proposal_actions = actions_from_logits_argmax(logits1, feats, H, W)
                elif u < args.p_model + args.p_noised:
                    # Noised expert: flip noise_p fraction of agent actions to random.
                    proposal_actions = expert.clone()
                    noise_mask = torch.rand_like(proposal_actions, dtype=torch.float) < args.noise_p
                    rand_acts = torch.randint(0, 5, proposal_actions.shape, device=device, dtype=torch.long)
                    proposal_actions = torch.where(noise_mask, rand_acts, proposal_actions)
                else:
                    proposal_actions = expert.clone()

                # Build action map and conflict map per batch element.
                action_maps = []
                conflict_maps = []
                for b in range(B):
                    am, cm = build_action_map_and_conflicts(
                        feats[b], proposal_actions[b], H, W, device=device,
                    )
                    action_maps.append(am)
                    conflict_maps.append(cm)
                A_prev = torch.stack(action_maps, dim=0)                   # [B, 5, H, W]
                C_prev = torch.stack(conflict_maps, dim=0)                 # [B, 11, H, W]

            # ── pass 2 ─────────────────────────────────────────────────────
            logits2, _ = policy(feats, prev_action_map=A_prev, prev_conflict_map=C_prev)
            pa2, t2 = per_agent_logits_at_positions(logits2, feats, expert)
            loss2 = F.cross_entropy(pa2, t2)

            loss = args.alpha * loss1 + args.beta * loss2

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
            optimizer.step()

            # ── stats ──────────────────────────────────────────────────────
            with torch.no_grad():
                acc1 = (pa1.argmax(dim=-1) == t1).float().mean().item()
                acc2 = (pa2.argmax(dim=-1) == t2).float().mean().item()
            epoch_loss += loss.item()
            epoch_loss1 += loss1.item()
            epoch_loss2 += loss2.item()
            epoch_acc1 += acc1
            epoch_acc2 += acc2
            n_acc += 1
            global_step += 1

            if (batch_i + 1) % args.log_interval == 0:
                print(f"  e{epoch+1} b{batch_i+1}/{n_batches}  "
                      f"loss={loss.item():.4f} (l1={loss1.item():.4f}  l2={loss2.item():.4f})  "
                      f"acc1={acc1:.3f}  acc2={acc2:.3f}")

        # ── validation ─────────────────────────────────────────────────────
        policy.eval()
        val_acc1_sum = 0.0; val_acc2_sum = 0.0; val_n = 0
        with torch.no_grad():
            for vi in range(0, len(data_val), args.batch_size):
                vbatch = data_val[vi : vi + args.batch_size]
                feats_v = torch.stack([x[0] for x in vbatch], dim=0).to(device)
                expert_v = torch.stack([x[1] for x in vbatch], dim=0).long().to(device)
                logits1, _ = policy(feats_v)
                pa1v, t1v = per_agent_logits_at_positions(logits1, feats_v, expert_v)
                if pa1v is None:
                    continue
                # For val pass 2, use model's first-pass argmax (most realistic).
                prop_v = actions_from_logits_argmax(logits1, feats_v, H, W)
                A_v_list = []; C_v_list = []
                for b in range(feats_v.shape[0]):
                    am, cm = build_action_map_and_conflicts(feats_v[b], prop_v[b], H, W, device=device)
                    A_v_list.append(am); C_v_list.append(cm)
                A_v = torch.stack(A_v_list, dim=0)
                C_v = torch.stack(C_v_list, dim=0)
                logits2, _ = policy(feats_v, prev_action_map=A_v, prev_conflict_map=C_v)
                pa2v, t2v = per_agent_logits_at_positions(logits2, feats_v, expert_v)
                val_acc1_sum += (pa1v.argmax(dim=-1) == t1v).float().sum().item()
                val_acc2_sum += (pa2v.argmax(dim=-1) == t2v).float().sum().item()
                val_n += t1v.numel()
        val_acc1 = val_acc1_sum / max(1, val_n)
        val_acc2 = val_acc2_sum / max(1, val_n)

        dt = time.time() - ep_t0
        print(f"[epoch {epoch+1}/{args.epochs}] "
              f"loss={epoch_loss/n_acc:.4f}  l1={epoch_loss1/n_acc:.4f}  l2={epoch_loss2/n_acc:.4f}  "
              f"train_acc1={epoch_acc1/n_acc:.3f}  train_acc2={epoch_acc2/n_acc:.3f}  "
              f"VAL_acc1={val_acc1:.3f}  VAL_acc2={val_acc2:.3f}  ({dt:.1f}s)")

        # Save checkpoint
        ckpt_path = os.path.join(save_dir, f"epoch_{epoch+1:02d}.pt")
        torch.save({"unet": policy.state_dict()}, ckpt_path)
        # Track best by val acc2 (refinement output)
        if val_acc2 > best_val_acc2:
            best_val_acc2 = val_acc2
            torch.save({"unet": policy.state_dict()}, os.path.join(save_dir, "best.pt"))
            epochs_since_best = 0
            print(f"  ← new best val_acc2={val_acc2:.3f}, saved best.pt")
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.early_stop_patience:
                print(f"  Early stopping (no improvement for {args.early_stop_patience} epochs)")
                break

    final_path = os.path.join(save_dir, "final.pt")
    torch.save({"unet": policy.state_dict()}, final_path)
    print(f"\nSaved final model to {final_path}")


if __name__ == "__main__":
    main()
