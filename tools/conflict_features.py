"""Conflict feature computation for iterative joint-action refinement.

Given a proposed joint action, deterministically compute conflict feedback
features that tell the next refinement pass where the proposal is inconsistent.

Inputs (per environment state, per refinement round):
  - positions [N, 2]              : agent (row, col)
  - actions   [N]                 : proposed action per agent (0=stay, 1=R, 2=L, 3=U, 4=D)
  - obstacle  [H, W]              : 1=wall, 0=free
  - H, W                          : grid dimensions

Output: conflict feature map [11, H, W] consisting of:
  channels 0-4: per-action one-hot at agent's current cell (action proposal)
  channel 5:    proposed next-cell occupancy count (normalized by N)
  channel 6:    vertex-conflict indicator (1 at cells where >1 agent wants to move)
  channel 7:    edge-swap conflict indicator (1 at cells involved in swap)
  channel 8:    "this agent is in any conflict" indicator at agent's current cell
  channel 9:    "proposed target is an obstacle" indicator at agent's current cell
  channel 10:   confidence (entropy of action distribution) — optional, zero if logits None

NOTE: feedback is keyed on the AGENT'S CURRENT CELL so the network reads it at
the same spatial location where it later reads action logits at refinement r+1.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Optional

# RAILGUN action encoding: 0=stay, 1=right, 2=left, 3=up, 4=down
ACTION_DELTAS = np.array([(0, 0), (0, 1), (0, -1), (-1, 0), (1, 0)], dtype=np.int64)


def compute_conflict_features(
    positions: torch.Tensor,
    actions: torch.Tensor,
    obstacle: torch.Tensor,
    H: int,
    W: int,
    logits: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return conflict feature map [11, H, W].

    positions : [N, 2] long  — agent (row, col)
    actions   : [N]    long  — chosen action per agent
    obstacle  : [H, W] float — 1=wall, 0=free
    logits    : [N, 5] float (optional) — for entropy channel

    All inputs on the same device. Output on the same device.
    """
    device = positions.device
    N = positions.shape[0]
    feat = torch.zeros(11, H, W, device=device, dtype=torch.float32)
    if N == 0:
        return feat

    r = positions[:, 0].long().clamp(0, H - 1)
    c = positions[:, 1].long().clamp(0, W - 1)
    a = actions.long().clamp(0, 4)

    # Channels 0-4: action one-hot at agent's current cell.
    for action_idx in range(5):
        mask = (a == action_idx)
        if mask.any():
            feat[action_idx, r[mask], c[mask]] = 1.0

    # Compute proposed next positions (with obstacle/edge clamp).
    deltas_t = torch.tensor(ACTION_DELTAS, device=device, dtype=torch.long)  # [5, 2]
    delta = deltas_t[a]                                                       # [N, 2]
    nr = (r + delta[:, 0]).clamp(0, H - 1)
    nc = (c + delta[:, 1]).clamp(0, W - 1)

    # Mark whether the proposed target is an obstacle (channel 9 at AGENT's current cell)
    obs_at_target = obstacle[nr, nc].float()
    feat[9, r, c] = obs_at_target

    # Channel 5: proposed next-cell occupancy count (at the TARGET cell).
    # Accumulate counts via scatter_add.
    flat_target = nr * W + nc                                                 # [N]
    counts_flat = torch.zeros(H * W, device=device, dtype=torch.float32)
    counts_flat.scatter_add_(0, flat_target, torch.ones(N, device=device))
    counts_map = counts_flat.view(H, W)
    feat[5] = counts_map / max(float(N), 1.0)

    # Channel 6: vertex-conflict — cells where >1 agents target the same cell.
    vertex_conflict_at_target = (counts_map > 1.0).float()
    # Mark at AGENT's current cell (so network reads it where it reads logits).
    feat[6, r, c] = vertex_conflict_at_target[nr, nc]

    # Channel 7: edge-swap conflict — agent i moving to j's current cell AND
    # agent j moving to i's current cell.
    # Build a mapping cell -> agent_index_at_cell.
    agent_at_cell = -torch.ones(H * W, device=device, dtype=torch.long)
    flat_curr = r * W + c
    agent_at_cell[flat_curr] = torch.arange(N, device=device)
    # For each agent i, who's at agent i's target cell?
    other_at_target = agent_at_cell[flat_target]                              # [N], -1 if no one
    edge_swap = torch.zeros(N, dtype=torch.bool, device=device)
    has_other = other_at_target >= 0
    if has_other.any():
        idx_other = other_at_target[has_other].clamp(min=0)
        # Did THAT agent move back to my current cell?
        other_target_flat = nr[idx_other] * W + nc[idx_other]
        my_curr_flat = flat_curr[has_other]
        swap_mask = (other_target_flat == my_curr_flat) & (idx_other != torch.arange(N, device=device)[has_other])
        edge_swap[has_other] = swap_mask
    feat[7, r[edge_swap], c[edge_swap]] = 1.0

    # Channel 8: "this agent is in any conflict" — vertex OR edge-swap OR target-is-obstacle.
    in_conflict = (
        (vertex_conflict_at_target[nr, nc] > 0)
        | edge_swap
        | (obs_at_target > 0.5)
    )
    feat[8, r[in_conflict], c[in_conflict]] = 1.0

    # Channel 10: optional entropy (confidence) of action distribution.
    if logits is not None:
        # logits [N, 5] → entropy scalar per agent
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        ent = -(probs * log_probs).sum(dim=-1)                                # [N]
        ent_norm = ent / float(np.log(5))                                     # in [0, 1]
        feat[10, r, c] = ent_norm

    return feat


def _smoke_test():
    """Quick test on a tiny synthetic state."""
    H, W = 8, 8
    obstacle = torch.zeros(H, W)
    obstacle[3, 3] = 1  # one wall
    # 4 agents, two heading to same cell (vertex conflict),
    # two swapping (edge conflict), one moving into wall.
    positions = torch.tensor([
        [0, 0],   # agent 0
        [0, 2],   # agent 1
        [5, 5],   # agent 2
        [5, 6],   # agent 3
    ], dtype=torch.long)
    # actions: 0 right, 1 left → both target (0,1). Vertex conflict at (0,1)?
    # Actually: agent 0 at (0,0) right → (0,1). Agent 1 at (0,2) left → (0,1). Yes, vertex conflict.
    # Agent 2 at (5,5) right → (5,6). Agent 3 at (5,6) left → (5,5). Edge swap.
    # No agents moving into walls here; add one: agent at (3,2) right → (3,3) wall.
    positions = torch.cat([positions, torch.tensor([[3, 2]], dtype=torch.long)], dim=0)
    actions = torch.tensor([1, 2, 1, 2, 1], dtype=torch.long)
    feat = compute_conflict_features(positions, actions, obstacle, H, W)
    print(f"conflict feature shape: {tuple(feat.shape)}")
    # Vertex conflict cells: agent 0 and 1 at their current cells (0,0) and (0,2)
    print(f"  vertex conflict at (0,0): {feat[6, 0, 0].item()}  expect 1.0")
    print(f"  vertex conflict at (0,2): {feat[6, 0, 2].item()}  expect 1.0")
    print(f"  edge swap at (5,5):       {feat[7, 5, 5].item()}  expect 1.0")
    print(f"  edge swap at (5,6):       {feat[7, 5, 6].item()}  expect 1.0")
    print(f"  in_conflict at (0,0):     {feat[8, 0, 0].item()}  expect 1.0")
    print(f"  obstacle target at (3,2): {feat[9, 3, 2].item()}  expect 1.0")
    print(f"  action ch (right=1) at (0,0): {feat[1, 0, 0].item()}  expect 1.0")


if __name__ == "__main__":
    _smoke_test()
