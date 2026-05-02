"""POGEMA environment wrapper that produces RAILGUN-compatible [6, H, W] feature tensors.

Feature channels (identical to construct_features_native.cpp / tools/utils.py):
  [0] obstacle map   (1=wall, 0=free)
  [1] agent positions (value = 1-based agent_id, 0=empty)
  [2] goal  positions (value = 1-based agent_id, 0=empty)
  [3] distance-to-goal (BFS from goal, normalised by H+W)
  [4] gradient_x (finite difference of distance map, row axis)
  [5] gradient_y (finite difference of distance map, col axis)

Action convention (RAILGUN canonical):
  0: stay   [ 0,  0]
  1: right  [ 0, +1]
  2: left   [ 0, -1]
  3: up     [-1,  0]
  4: down   [+1,  0]

POGEMA MOVES (confirmed empirically):
  0: stay, 1: up[-1,0], 2: down[+1,0], 3: left[0,-1], 4: right[0,+1]
"""

import glob
import os
import random
from collections import deque
from typing import Optional

import numpy as np
import torch
from pogema import GridConfig, pogema_v0

# RAILGUN action index → POGEMA action index
# Verified against pogema GridConfig.MOVES = [[0,0],[-1,0],[1,0],[0,-1],[0,1]]
RAILGUN_TO_POGEMA = {
    0: 0,  # stay  → stay
    1: 4,  # right → POGEMA 4  [0,+1]
    2: 3,  # left  → POGEMA 3  [0,-1]
    3: 1,  # up    → POGEMA 1  [-1, 0]
    4: 2,  # down  → POGEMA 2  [+1, 0]
}

# Default map directory relative to this file's RAILGUN root
_DEFAULT_MAP_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "map_files"
)

NOT_FOUND_DIST = 2048


def load_map_file(path: str) -> np.ndarray:
    """Parse a MovingAI .map file and return a binary obstacle array.

    Returns
    -------
    np.ndarray  shape [H, W], dtype uint8
        1 = obstacle, 0 = free
    """
    with open(path) as f:
        lines = f.readlines()
    # Find 'map' keyword line
    map_start = next(i for i, l in enumerate(lines) if l.strip() == "map") + 1
    grid_lines = [l.rstrip("\n") for l in lines[map_start:]]
    grid = []
    for row in grid_lines:
        if not row:
            continue
        grid.append([0 if c == "." else 1 for c in row])
    return np.array(grid, dtype=np.uint8)


def collect_map_files(map_dir: str) -> list[str]:
    """Return all .map files found recursively under map_dir."""
    return sorted(glob.glob(os.path.join(map_dir, "**", "*.map"), recursive=True))


def bfs_distance_from_goal(obstacle: np.ndarray, goal_rc: tuple[int, int]) -> np.ndarray:
    """BFS distance map from every free cell to goal_rc.

    Returns
    -------
    np.ndarray shape [H, W], dtype float32
        dist[r, c] = BFS steps from (r,c) to goal; NOT_FOUND_DIST if unreachable.
    """
    H, W = obstacle.shape
    dist = np.full((H, W), NOT_FOUND_DIST, dtype=np.float32)
    gr, gc = goal_rc
    if obstacle[gr, gc]:
        return dist
    dist[gr, gc] = 0.0
    q = deque()
    q.append((gr, gc))
    while q:
        r, c = q.popleft()
        d = dist[r, c]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not obstacle[nr, nc] and dist[nr, nc] == NOT_FOUND_DIST:
                dist[nr, nc] = d + 1.0
                q.append((nr, nc))
    return dist


class POGEMARailgunEnv:
    """POGEMA wrapper that returns RAILGUN-compatible [6, H, W] feature tensors.

    Parameters
    ----------
    map_source : str | list[str] | np.ndarray | None
        - str path to a directory: randomly sample a .map file each episode
        - str path to a single .map file: use that map every episode
        - list of str file paths: sample uniformly from list each episode
        - np.ndarray [H, W] uint8: use that fixed obstacle grid every episode
        - None: fall back to GridConfig random-map generation (density/size required)
    num_agents : int
    max_steps : int
    seed : int | None
    density : float
        Only used when map_source is None (random POGEMA maps).
    size : int
        Only used when map_source is None.
    """

    def __init__(
        self,
        map_source=None,
        num_agents: int = 8,
        max_steps: int = 256,
        seed: Optional[int] = None,
        density: float = 0.3,
        size: int = 16,
    ):
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.seed = seed
        self.density = density
        self.size = size

        # Resolve map source
        if isinstance(map_source, np.ndarray):
            self._map_files = None
            self._fixed_map = map_source
        elif isinstance(map_source, str):
            if os.path.isdir(map_source):
                self._map_files = collect_map_files(map_source)
                if not self._map_files:
                    raise ValueError(f"No .map files found in {map_source}")
                self._fixed_map = None
            elif os.path.isfile(map_source):
                self._map_files = [map_source]
                self._fixed_map = None
            else:
                raise ValueError(f"map_source path does not exist: {map_source}")
        elif isinstance(map_source, list):
            self._map_files = map_source
            self._fixed_map = None
        else:
            # Random POGEMA maps
            self._map_files = None
            self._fixed_map = None

        self._env = None
        self._obstacle: Optional[np.ndarray] = None   # [H, W] current episode map
        self._dist_maps: Optional[list] = None         # per-agent BFS dist arrays
        self._goal_positions: Optional[list] = None    # [(r,c), ...]
        self._step_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> tuple[torch.Tensor, dict]:
        """Reset environment and return initial feature tensor."""
        rng_seed = seed if seed is not None else (self.seed if self.seed is not None else random.randint(0, 2**31))

        obstacle_map_arg = self._sample_obstacle_map()
        gc = self._make_grid_config(obstacle_map_arg, rng_seed)
        self._env = pogema_v0(gc)
        self._env.reset()

        inner = self._env.unwrapped
        obs_radius = inner.grid.config.obs_radius
        full_obs = inner.get_obstacles()  # padded [H+2p, W+2p]
        H = gc.height
        W = gc.width
        self._obstacle = full_obs[obs_radius:obs_radius + H, obs_radius:obs_radius + W].astype(np.uint8)

        self._update_goals_and_dists()
        self._step_count = 0

        feat = self._build_feature()
        return feat, {}

    def step(self, actions: np.ndarray) -> tuple[torch.Tensor, np.ndarray, bool, dict]:
        """Step the environment.

        Parameters
        ----------
        actions : np.ndarray [N] int
            RAILGUN action indices (0=stay, 1=right, 2=left, 3=up, 4=down).

        Returns
        -------
        feature : Tensor [6, H, W]
        rewards : np.ndarray [N]
        done : bool
        info : dict
        """
        assert self._env is not None, "call reset() before step()"

        # Record previous distances for reward computation
        prev_dists = self._agent_distances()

        # Remap RAILGUN actions → POGEMA actions
        pogema_actions = [RAILGUN_TO_POGEMA[int(a)] for a in actions]

        obs, rews, terms, truncs, infos = self._env.step(pogema_actions)
        self._step_count += 1

        # Update goal positions if any agent reached its goal (lifelong: goals reassigned)
        self._update_goals_and_dists()

        curr_dists = self._agent_distances()
        inner = self._env.unwrapped
        reached = np.array(inner.grid.on_goal, dtype=bool) if hasattr(inner.grid, "on_goal") else np.zeros(self.num_agents, dtype=bool)

        rewards = self._compute_rewards(prev_dists, curr_dists, reached)
        done = all(terms) or all(truncs)
        feat = self._build_feature()
        return feat, rewards, done, {}

    @property
    def obstacle(self) -> np.ndarray:
        """Current episode obstacle grid [H, W] uint8."""
        return self._obstacle

    @property
    def map_shape(self) -> tuple[int, int]:
        if self._obstacle is not None:
            return self._obstacle.shape
        return (self.size, self.size)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_obstacle_map(self) -> Optional[np.ndarray]:
        if self._fixed_map is not None:
            return self._fixed_map
        if self._map_files is not None:
            path = random.choice(self._map_files)
            return load_map_file(path)
        return None  # POGEMA will generate randomly

    def _make_grid_config(self, obstacle: Optional[np.ndarray], seed: int) -> GridConfig:
        if obstacle is not None:
            H, W = obstacle.shape
            # Convert to list-of-strings format POGEMA expects
            map_rows = ["".join("@" if obstacle[r, c] else "." for c in range(W)) for r in range(H)]
            return GridConfig(
                map=map_rows,
                num_agents=self.num_agents,
                seed=seed,
                max_episode_steps=self.max_steps,
                observation_type="default",
            )
        else:
            return GridConfig(
                num_agents=self.num_agents,
                size=self.size,
                density=self.density,
                seed=seed,
                max_episode_steps=self.max_steps,
                observation_type="default",
            )

    def _update_goals_and_dists(self):
        """Refresh goal positions and BFS distance maps from current POGEMA state."""
        inner = self._env.unwrapped
        obs_radius = inner.grid.config.obs_radius
        targets_padded = inner.get_targets_xy()  # list of (row_pad, col_pad)
        H, W = self._obstacle.shape

        self._goal_positions = [
            (t[0] - obs_radius, t[1] - obs_radius) for t in targets_padded
        ]
        self._dist_maps = [
            bfs_distance_from_goal(self._obstacle, goal_rc)
            for goal_rc in self._goal_positions
        ]

    def _agent_positions(self) -> list[tuple[int, int]]:
        """Agent positions in unpadded map coordinates."""
        inner = self._env.unwrapped
        obs_radius = inner.grid.config.obs_radius
        padded = inner.get_agents_xy()
        return [(p[0] - obs_radius, p[1] - obs_radius) for p in padded]

    def _agent_distances(self) -> np.ndarray:
        """BFS distance to goal for each agent."""
        positions = self._agent_positions()
        dists = np.zeros(self.num_agents, dtype=np.float32)
        for i, (r, c) in enumerate(positions):
            dm = self._dist_maps[i]
            H, W = dm.shape
            r_c = max(0, min(r, H - 1))
            c_c = max(0, min(c, W - 1))
            dists[i] = dm[r_c, c_c]
        return dists

    def _build_feature(self) -> torch.Tensor:
        """Construct [6, H, W] RAILGUN feature tensor from current POGEMA state."""
        H, W = self._obstacle.shape
        feat = torch.zeros(6, H, W, dtype=torch.float32)

        # Channel 0: obstacle
        feat[0] = torch.from_numpy(self._obstacle.astype(np.float32))

        positions = self._agent_positions()
        normaliser = float(H + W)

        for i, (r, c) in enumerate(positions):
            agent_id = float(i + 1)
            r_c = max(0, min(r, H - 1))
            c_c = max(0, min(c, W - 1))

            # Channel 1: agent position
            feat[1, r_c, c_c] = agent_id

            # Channel 2: goal position
            gr, gc_ = self._goal_positions[i]
            gr = max(0, min(gr, H - 1))
            gc_ = max(0, min(gc_, W - 1))
            feat[2, gr, gc_] = agent_id

            # Channel 3: BFS distance to goal (normalised)
            dm = self._dist_maps[i]
            d = float(dm[r_c, c_c])
            feat[3, r_c, c_c] = d / normaliser if d < NOT_FOUND_DIST else 0.0

            # Channels 4-5: finite-difference gradient of distance map
            # gradient_x (row direction): d[r-1,c] - d[r+1,c] / 2
            # gradient_y (col direction): d[r,c-1] - d[r,c+1] / 2
            d_up   = float(dm[max(r_c - 1, 0), c_c])
            d_down = float(dm[min(r_c + 1, H - 1), c_c])
            d_left = float(dm[r_c, max(c_c - 1, 0)])
            d_right = float(dm[r_c, min(c_c + 1, W - 1)])

            # Treat unreachable cells as same distance (gradient = 0 toward them)
            if d_up >= NOT_FOUND_DIST:    d_up = d
            if d_down >= NOT_FOUND_DIST:  d_down = d
            if d_left >= NOT_FOUND_DIST:  d_left = d
            if d_right >= NOT_FOUND_DIST: d_right = d

            gx = (d_up - d_down) / (2.0 * normaliser)
            gy = (d_left - d_right) / (2.0 * normaliser)
            feat[4, r_c, c_c] = gx
            feat[5, r_c, c_c] = gy

        return feat

    def _compute_rewards(
        self,
        prev_dists: np.ndarray,
        curr_dists: np.ndarray,
        reached: np.ndarray,
    ) -> np.ndarray:
        """
        r = +10.0 * reached_goal
            + 0.5  * (bfs_dist decreased)
            - 0.1  (time penalty per step)
        """
        rewards = np.full(self.num_agents, -0.1, dtype=np.float32)
        rewards += 10.0 * reached.astype(np.float32)
        improved = (prev_dists - curr_dists > 0) & ~reached
        rewards += 0.5 * improved.astype(np.float32)
        return rewards
