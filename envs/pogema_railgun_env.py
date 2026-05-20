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


def _trinary_gradient(delta_a: float, delta_b: float) -> int:
    """Mirrors RAILGUN's C++ feature builder branching for the gradient channels.

    Inputs are signed deltas (neighbour_distance - current_distance) for the two
    candidate moves along an axis. Returns -1, 0, or +1 with random tie-breaking
    matching the C++ logic. The exact semantic mapping of {-1, 0, +1} to spatial
    directions depends on which two deltas are passed and is what the pretrained
    model learned during training.

    See RAILGUN/tools/extensions/construct_features_native.cpp lines 270-306.
    """
    if delta_a > 0 and delta_b > 0:
        return 0
    if delta_a >= 0 and delta_b < 0:
        return 1
    if delta_a < 0 and delta_b >= 0:
        return -1
    if delta_a < 0 and delta_b < 0:
        return random.choice((-1, 1))
    if delta_a == 0 and delta_b == 0:
        return random.choice((-1, 0, 1))
    if delta_a == 0 and delta_b > 0:
        return random.choice((-1, 0))
    if delta_a > 0 and delta_b == 0:
        return random.choice((0, 1))
    return random.choice((-1, 1))


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
        feature_type: str = "none",
        reward_mode: str = "dense_selfish",
    ):
        if feature_type not in ("gradient", "none"):
            raise ValueError(f"feature_type must be 'gradient' or 'none', got {feature_type!r}")
        if reward_mode not in ("dense_selfish", "sparse", "cooperative", "cooperative_v2"):
            raise ValueError(
                f"reward_mode must be 'dense_selfish' | 'sparse' | 'cooperative' | 'cooperative_v2', "
                f"got {reward_mode!r}"
            )
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.seed = seed
        self.density = density
        self.size = size
        self.reward_mode = reward_mode
        # Track prior-reached status across steps (for "first time reached" bonus
        # in cooperative / cooperative_v2 modes). Reset in reset().
        self._prev_reached: Optional[np.ndarray] = None
        # 'none'     → channels 4/5 are signed (goal_row-agent_row, goal_col-agent_col)
        #              displacement. This matches the upstream RAILGUN released
        #              checkpoint's training distribution; default.
        # 'gradient' → channels 4/5 are trinary {-1, 0, +1} (matches
        #              construct_features_native.cpp). Use only when training
        #              new models against that scheme.
        self.feature_type = feature_type

        # Resolve map source
        self._mix_sources = None  # set only when map_source is a curriculum mix
        if isinstance(map_source, dict) and "mix" in map_source:
            # Curriculum mix: list of {weight: float, source: str|None} entries.
            # source = directory path → sample from map files; None/"random" → POGEMA random.
            mix = []
            for entry in map_source["mix"]:
                weight = float(entry["weight"])
                src = entry.get("source")
                if src is None or src == "random":
                    mix.append((weight, None))  # POGEMA random
                elif isinstance(src, str) and os.path.isdir(src):
                    files = collect_map_files(src)
                    if not files:
                        raise ValueError(f"No .map files in {src}")
                    mix.append((weight, files))
                elif isinstance(src, str) and os.path.isfile(src):
                    mix.append((weight, [src]))
                else:
                    raise ValueError(f"Invalid mix source: {src}")
            self._mix_sources = mix
            self._map_files = None
            self._fixed_map = None
        elif isinstance(map_source, np.ndarray):
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
        self._prev_reached = np.zeros(self.num_agents, dtype=bool)

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

        # Record previous state for reward computation
        prev_dists = self._agent_distances()
        prev_positions = np.array(self._agent_positions(), dtype=np.int64)

        # Remap RAILGUN actions → POGEMA actions
        pogema_actions = [RAILGUN_TO_POGEMA[int(a)] for a in actions]

        obs, rews, terms, truncs, infos = self._env.step(pogema_actions)
        self._step_count += 1

        # Update goal positions if any agent reached its goal (lifelong: goals reassigned)
        self._update_goals_and_dists()

        curr_dists = self._agent_distances()
        curr_positions = np.array(self._agent_positions(), dtype=np.int64)
        # Use POGEMA's per-agent termination flag (True = agent reached goal this step)
        reached = np.array(terms[:self.num_agents], dtype=bool)

        # Conflict detection: an agent is "conflicted" if it attempted a non-stay
        # action but its position didn't change. (Wall block or PIBT-style revert.)
        attempted_move = np.array([RAILGUN_TO_POGEMA[int(a)] != 0 for a in actions], dtype=bool)
        stayed_in_place = np.all(curr_positions == prev_positions, axis=1)
        conflicts = attempted_move & stayed_in_place & ~reached

        rewards = self._compute_rewards(prev_dists, curr_dists, reached, conflicts, actions=actions)
        # Episode ends when all agents finish (terms) or time limit hit (any truncs)
        done = all(terms) or any(truncs)
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
        if self._mix_sources is not None:
            weights = [w for w, _ in self._mix_sources]
            files = random.choices(self._mix_sources, weights=weights, k=1)[0][1]
            if files is None:
                return None  # POGEMA random
            return load_map_file(random.choice(files))
        if self._fixed_map is not None:
            return self._fixed_map
        if self._map_files is not None:
            path = random.choice(self._map_files)
            return load_map_file(path)
        return None  # POGEMA will generate randomly

    def _make_grid_config(self, obstacle: Optional[np.ndarray], seed: int) -> GridConfig:
        if obstacle is not None:
            H, W = obstacle.shape
            # POGEMA's str_map_to_list: '#' is obstacle, '.' is free, '@' is a
            # "possible agent location" marker that resolves to FREE — using
            # '@' here would silently make every wall walkable in the env.
            map_str = "\n".join(
                "".join("#" if obstacle[r, c] else "." for c in range(W))
                for r in range(H)
            )
            return GridConfig(
                map=map_str,
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
        """Construct [6, H, W] feature tensor for the RAILGUN UNet.

        Channels 0–3 are identical across both feature_type modes:
          [0] obstacle                  (1=wall, 0=free)
          [1] agent_id at agent cell    (1-based)
          [2] agent_id at goal cell     (1-based)
          [3] raw BFS distance from goal at the agent's cell  (2048 = unreachable)

        Channels 4, 5 depend on self.feature_type:
          'none'     → signed (goal_row - agent_row, goal_col - agent_col).
                       This is what the upstream RAILGUN released checkpoint
                       was trained on — empirically the only scheme under
                       which it produces sensible argmax actions.
          'gradient' → trinary {-1, 0, +1} from delta-distance comparisons,
                       matching construct_features_native.cpp exactly. Use
                       only when training new models against that scheme.
        """
        H, W = self._obstacle.shape
        feat = torch.zeros(6, H, W, dtype=torch.float32)

        # Channel 0: obstacle
        feat[0] = torch.from_numpy(self._obstacle.astype(np.float32))

        positions = self._agent_positions()

        # Channels 1, 2 first so the gradient-mode validity check can read them.
        for i, (r, c) in enumerate(positions):
            r_c = max(0, min(r, H - 1))
            c_c = max(0, min(c, W - 1))
            feat[1, r_c, c_c] = float(i + 1)
            gr, gc_ = self._goal_positions[i]
            gr = max(0, min(gr, H - 1))
            gc_ = max(0, min(gc_, W - 1))
            feat[2, gr, gc_] = float(i + 1)

        # Channels 3, 4, 5
        for i, (r, c) in enumerate(positions):
            r_c = max(0, min(r, H - 1))
            c_c = max(0, min(c, W - 1))

            dm = self._dist_maps[i]
            d_self = float(dm[r_c, c_c])
            feat[3, r_c, c_c] = d_self if d_self < NOT_FOUND_DIST else float(NOT_FOUND_DIST)

            if self.feature_type == "none":
                gr, gc_ = self._goal_positions[i]
                feat[4, r_c, c_c] = float(gr - r_c)
                feat[5, r_c, c_c] = float(gc_ - c_c)
                continue

            # feature_type == "gradient"
            def _valid(nr: int, nc: int) -> bool:
                if not (0 <= nr < H and 0 <= nc < W):
                    return False
                if self._obstacle[nr, nc] != 0:
                    return False
                if feat[1, nr, nc].item() != 0:  # cell occupied by some agent
                    return False
                return True

            left_dist  = float(dm[r_c - 1, c_c]) if _valid(r_c - 1, c_c) else float(NOT_FOUND_DIST)
            right_dist = float(dm[r_c + 1, c_c]) if _valid(r_c + 1, c_c) else float(NOT_FOUND_DIST)
            up_dist    = float(dm[r_c, c_c - 1]) if _valid(r_c, c_c - 1) else float(NOT_FOUND_DIST)
            down_dist  = float(dm[r_c, c_c + 1]) if _valid(r_c, c_c + 1) else float(NOT_FOUND_DIST)

            feat[4, r_c, c_c] = float(_trinary_gradient(left_dist - d_self, right_dist - d_self))
            feat[5, r_c, c_c] = float(_trinary_gradient(down_dist - d_self, up_dist - d_self))

        return feat

    def _compute_rewards(
        self,
        prev_dists: np.ndarray,
        curr_dists: np.ndarray,
        reached: np.ndarray,
        conflicts: Optional[np.ndarray] = None,
        actions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Three reward modes selected by self.reward_mode:

        'dense_selfish' (default, matches iter-7 winner reward on maze-34):
            r = -0.1 + 10·reached + 0.5·improved_dist
            Per-agent dense distance reward. Rewards selfish progress; no
            explicit cooperative signal. Baseline.

        'sparse' (PhD student's proposal):
            r_i = 0.5 + 0.5·(remaining_time/limit)  if all agents reached
                  0                                  otherwise
            Episode-terminal binary. Cooperative by construction (all-or-nothing)
            but compute-sparse; PPO struggles with the credit assignment at 32
            agents.

        'cooperative' (our design):
            r_i = +0.02·(prev_dist_i - curr_dist_i)     # small dense progress
                  - 0.1 ·conflicted_step_i                # conflict penalty (targets
                                                          #   the selfish-argmax failure mode)
                  + 0.3 ·just_reached_first_time_i        # per-agent first-reach bonus
                  + (0.5 + 0.5·remaining/limit) [if all reached, applied to all agents]
            Combines dense gradient with cooperative sparse terminal. Designed
            to teach RAILGUN's per-cell argmax to avoid joint-action conflicts.

        'cooperative_v2' (potential-based shaping, Ng et al. 1999):
            Team-progress potential φ(s) = - (1/N) · Σ_i min(dist_i, MAX_DIST)
            r_shape = φ(s') − φ(s)   (shared across all agents — policy-invariant)
            r_i = r_shape
                  + 0.3·just_reached_first_time_i       # per-agent reach bonus
                  + (0.5 + 0.5·remaining/limit)·[all_solved]  # team terminal
            Replaces the conflict penalty with a principled shaping signal that
            naturally rewards cooperative behaviour: agents that yield to let
            teammates progress get a positive shaped reward (team-distance dropped).
            No "stay" pathology because there is no fixed per-step penalty.
        """
        N = self.num_agents
        rewards = np.zeros(N, dtype=np.float32)

        if self.reward_mode == "dense_selfish":
            rewards.fill(-0.1)
            rewards += 10.0 * reached.astype(np.float32)
            improved = (prev_dists - curr_dists > 0) & ~reached
            rewards += 0.5 * improved.astype(np.float32)
            return rewards

        if self.reward_mode == "sparse":
            if bool(np.all(reached)):
                time_remaining = max(0, self.max_steps - self._step_count)
                bonus = 0.5 + 0.5 * (time_remaining / float(self.max_steps))
                rewards.fill(np.float32(bonus))
            return rewards

        if self.reward_mode == "cooperative":
            # Dense per-agent distance reduction (small).
            delta = (prev_dists - curr_dists).astype(np.float32)
            rewards += 0.02 * delta
            # Conflict penalty: targets the selfish-argmax failure mode.
            if conflicts is not None:
                rewards -= 0.1 * conflicts.astype(np.float32)
            # Per-agent first-time-reached bonus.
            if self._prev_reached is None:
                self._prev_reached = np.zeros(N, dtype=bool)
            just_reached = reached & ~self._prev_reached
            rewards += 0.3 * just_reached.astype(np.float32)
            self._prev_reached = reached.copy()
            # Cooperative terminal bonus.
            if bool(np.all(reached)):
                time_remaining = max(0, self.max_steps - self._step_count)
                bonus = 0.5 + 0.5 * (time_remaining / float(self.max_steps))
                rewards += np.float32(bonus)
            return rewards

        # cooperative_v2 — potential-based team-progress shaping +
        # optional anti-stay bias (set SEAM_STAY_PENALTY env var, e.g. 0.02).
        # Anti-stay subtracts per-agent penalty when agent chose action 0 (stay)
        # while not at goal — counteracts the "stay is safe" equilibrium that
        # PPO/MAPPO get stuck in (freeze >80% across 8 prior variants).
        import os
        _stay_penalty = float(os.environ.get("SEAM_STAY_PENALTY", "0.0"))
        # Compute team potential at prev and curr states.
        # Clip distances at NOT_FOUND_DIST to avoid astronomical magnitudes on
        # unreachable cells (already cliped earlier by _agent_distances).
        prev_clip = np.minimum(prev_dists.astype(np.float32), float(NOT_FOUND_DIST))
        curr_clip = np.minimum(curr_dists.astype(np.float32), float(NOT_FOUND_DIST))
        # φ(s) = − (1/N) · Σ dist; r_shape = φ(s') − φ(s) = (Σ prev − Σ curr) / N
        r_shape = float((prev_clip - curr_clip).sum() / float(N))
        rewards.fill(np.float32(r_shape))
        # Per-agent first-time-reached bonus.
        if self._prev_reached is None:
            self._prev_reached = np.zeros(N, dtype=bool)
        just_reached = reached & ~self._prev_reached
        rewards += 0.3 * just_reached.astype(np.float32)
        self._prev_reached = reached.copy()
        # Cooperative terminal bonus.
        if bool(np.all(reached)):
            time_remaining = max(0, self.max_steps - self._step_count)
            bonus = 0.5 + 0.5 * (time_remaining / float(self.max_steps))
            rewards += np.float32(bonus)
        # Anti-stay penalty
        if _stay_penalty > 0 and actions is not None:
            stay_mask = (np.asarray(actions, dtype=np.int64) == 0) & (~reached)
            rewards -= np.float32(_stay_penalty) * stay_mask.astype(np.float32)
        return rewards
