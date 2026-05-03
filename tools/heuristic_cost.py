"""Heuristic cooperative cost function for SEAM.

Replaces the phi-GNN with three hand-engineered signals:
  1. Local density    — agents within L-inf radius 3 of each agent
  2. Bottleneck score — betweenness centrality of the map graph (cached per map)
  3. Directional conflict — another agent's heading points toward my cell

Produces phi_costs [N] matching the format expected by
tools.cost_shaping.apply_cost_shaping (same as PhiGNN output).

Callable interface (drop-in for PhiGNN in path_formation.py):
  adapter = HeuristicPhiAdapter(obstacle_map)
  phi_costs, None = adapter(feature, positions, prev_positions, agent_ids)
"""

import hashlib
from functools import lru_cache

import networkx as nx
import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ──────────────────────────────────────────────────────────────────────────────
DENSITY_RADIUS   = 3
W_DENSITY        = 2.0
W_BOTTLENECK     = 3.0
W_CONFLICT       = 5.0
MAX_COST         = 5.0   # same clamp as PhiGNN.max_cost

# Cache: map_hash → betweenness_grid tensor
_betweenness_cache: dict[str, torch.Tensor] = {}


# ──────────────────────────────────────────────────────────────────────────────
# Betweenness centrality
# ──────────────────────────────────────────────────────────────────────────────

def _map_hash(obstacle: np.ndarray) -> str:
    return hashlib.md5(obstacle.tobytes()).hexdigest()


def precompute_betweenness(obstacle: np.ndarray) -> torch.Tensor:
    """Build networkx graph from passable cells, compute betweenness centrality.

    Returns
    -------
    Tensor [H, W] float32  — normalised betweenness in [0, 1].
    Cached by map content hash; safe to call repeatedly for the same map.
    """
    key = _map_hash(obstacle)
    if key in _betweenness_cache:
        return _betweenness_cache[key]

    H, W = obstacle.shape
    G = nx.Graph()

    # Nodes: every passable cell
    for r in range(H):
        for c in range(W):
            if not obstacle[r, c]:
                G.add_node((r, c))

    # Edges: 4-connected neighbours
    for r in range(H):
        for c in range(W):
            if obstacle[r, c]:
                continue
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not obstacle[nr, nc]:
                    G.add_edge((r, c), (nr, nc))

    # Approximate betweenness (k=min(200, nodes)) for large maps
    n_nodes = G.number_of_nodes()
    k = min(200, n_nodes) if n_nodes > 200 else None
    bc_dict = nx.betweenness_centrality(G, k=k, normalized=True)

    grid = np.zeros((H, W), dtype=np.float32)
    for (r, c), val in bc_dict.items():
        grid[r, c] = val

    # Normalise so maximum = 1
    max_val = grid.max()
    if max_val > 0:
        grid /= max_val

    result = torch.from_numpy(grid)
    _betweenness_cache[key] = result
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Core heuristic
# ──────────────────────────────────────────────────────────────────────────────

def compute_heuristic_phi(
    positions: torch.Tensor,        # [N, 2] (row, col) — long
    prev_positions: torch.Tensor,   # [N, 2] positions at previous step
    betweenness_grid: torch.Tensor, # [H, W]
    w_density: float = W_DENSITY,
    w_bottleneck: float = W_BOTTLENECK,
    w_conflict: float = W_CONFLICT,
) -> tuple[torch.Tensor, dict]:  # ([N] float32, component dict)
    """Compute per-agent heuristic cooperative cost.

    Returns
    -------
    costs : Tensor [N]
        Clamped total phi cost per agent.
    components : dict
        Raw (pre-clamp) per-agent tensors for each signal:
        {"density": [N], "bottleneck": [N], "conflict": [N]}
    """
    N = positions.size(0)
    device = positions.device

    if N == 0:
        empty = torch.zeros(N, dtype=torch.float32, device=device)
        return empty, {"density": empty, "bottleneck": empty, "conflict": empty}

    pos_f = positions.float()

    # ── 1. Local density ────────────────────────────────────────────────────
    diff = pos_f.unsqueeze(0) - pos_f.unsqueeze(1)        # [N, N, 2]
    linf = diff.abs().max(dim=-1).values                    # [N, N]
    self_mask = ~torch.eye(N, dtype=torch.bool, device=device)
    in_radius = (linf <= DENSITY_RADIUS) & self_mask
    density_raw = in_radius.float().sum(dim=1)             # [N]
    density_cost = w_density * density_raw

    # ── 2. Bottleneck score ─────────────────────────────────────────────────
    rows = positions[:, 0].clamp(0, betweenness_grid.size(0) - 1)
    cols = positions[:, 1].clamp(0, betweenness_grid.size(1) - 1)
    bc = betweenness_grid.to(device)[rows, cols]            # [N], already normalized 0–1
    bottleneck_cost = w_bottleneck * bc

    # ── 3. Directional conflict ─────────────────────────────────────────────
    heading = positions.float() - prev_positions.float()    # [N, 2]
    conflict_cost = torch.zeros(N, dtype=torch.float32, device=device)

    heading_norm = heading.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # [N, 1]
    heading_unit = heading / heading_norm                               # [N, 2]

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dir_j_to_i = pos_f[i] - pos_f[j]                          # [2]
            dist_ji = dir_j_to_i.norm().clamp(min=1e-6)
            dir_j_to_i_unit = dir_j_to_i / dist_ji
            # How directly is j heading toward i?
            alignment = (heading_unit[j] * dir_j_to_i_unit).sum().clamp(0, 1)  # scalar
            # Proximity weight — closer agents contribute more
            proximity = (1.0 / (dist_ji + 1.0))
            conflict_cost[i] = conflict_cost[i] + w_conflict * alignment * proximity
            conflict_cost[j] = conflict_cost[j] + w_conflict * alignment * proximity  # aggressor also penalized

    costs = (density_cost + bottleneck_cost + conflict_cost).clamp(max=MAX_COST)
    components = {
        "density":    density_cost,
        "bottleneck": bottleneck_cost,
        "conflict":   conflict_cost,
    }
    return costs, components


# ──────────────────────────────────────────────────────────────────────────────
# Adapter (drop-in for PhiGNN in path_formation.py)
# ──────────────────────────────────────────────────────────────────────────────

class HeuristicPhiAdapter:
    """Wraps compute_heuristic_phi with the same call signature used by
    path_formation.py after the GNN removal:

        phi_costs, _ = phi_model(feature, positions, prev_positions, agent_ids)

    Parameters
    ----------
    obstacle : np.ndarray [H, W] uint8
        Obstacle map for the current episode (used to look up betweenness).
        Call update_map() when the map changes between episodes.
    """

    def __init__(
        self,
        obstacle: np.ndarray,
        w_density: float = W_DENSITY,
        w_bottleneck: float = W_BOTTLENECK,
        w_conflict: float = W_CONFLICT,
    ):
        self.w_density = w_density
        self.w_bottleneck = w_bottleneck
        self.w_conflict = w_conflict
        self._bc_grid: torch.Tensor = precompute_betweenness(obstacle)

    def update_map(self, obstacle: np.ndarray):
        """Update betweenness grid when the map changes (e.g. new episode)."""
        self._bc_grid = precompute_betweenness(obstacle)

    def __call__(
        self,
        feature: torch.Tensor,       # [C, H, W] — not used, kept for API compat
        positions: torch.Tensor,     # [N, 2]
        prev_positions: torch.Tensor,# [N, 2]
        agent_ids: torch.Tensor,     # [N] — not used, kept for API compat
    ) -> tuple[torch.Tensor, dict]:
        costs, components = compute_heuristic_phi(
            positions, prev_positions, self._bc_grid,
            w_density=self.w_density,
            w_bottleneck=self.w_bottleneck,
            w_conflict=self.w_conflict,
        )
        return costs, components
