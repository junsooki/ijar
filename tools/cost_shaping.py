"""Apply heuristic phi penalty to UNet logits — action-directional cost shaping.

For each agent i considering action a:
    The agent would land at  target = pos_i + DELTA[a].
    Subtract alpha * Σ_{j≠i} w(dist(target, pos_j)) * phi_j  from logit[a].

    w(d) = max(0, (R+1 - d) / (R+1))  where d = L-∞ distance

Effect: actions that move TOWARD high-phi neighbors get penalized most;
        actions that move AWAY get near-zero penalty.
        "Stay" is penalized based on dangerous agents at the current cell.

This is NOT shift-invariant — different actions get different penalties,
so the softmax distribution changes meaningfully.
"""

import torch

# RAILGUN action deltas: 0=stay, 1=right, 2=left, 3=up, 4=down
_DELTAS = torch.tensor([[0, 0], [0, 1], [0, -1], [-1, 0], [1, 0]], dtype=torch.float32)


def apply_cost_shaping(
    logits: torch.Tensor,      # [1, 5, H, W]
    phi_costs: torch.Tensor,   # [N]
    feat: torch.Tensor,        # [6, H, W] kept for API compat, not used
    positions: torch.Tensor,   # [N, 2] (row, col) — CPU long
    agent_ids: torch.Tensor,   # [N] — kept for API compat, not used
    alpha: float = 1.0,
    proximity_radius: int = 2,
    policy_aware: bool = False,
) -> torch.Tensor:
    """Return shaped logits [1, 5, H, W] (never modifies logits in-place).

    Each action gets an independent penalty based on where it would move the agent.

    Two modes for the per-action weight:

    - Distance-based (default): w(d) = max(0, (R+1-d)/(R+1)). Implicitly assumes
      a uniform-random prior over each neighbour's next move; w(d) decays linearly
      with L-inf distance.

    - Policy-aware (policy_aware=True): w_j(t_a) = π_j(action that lands j at t_a).
      Uses each neighbour's actual policy distribution from the current logits to
      compute the probability that j is at the target cell t_a next step. This is
      the principled posterior version: the prior for distance-based shaping was
      the average over uniform actions; here we use the actual policy. Gives 0
      for neighbours that can't reach t_a in one step (no policy mass there).
    """
    shaped = logits.clone()
    N = positions.size(0)
    if N == 0 or alpha == 0.0:
        return shaped

    dev    = logits.device
    pos_f  = positions.float().to(dev)          # [N, 2]
    phi    = phi_costs.float().to(dev)          # [N]
    R      = proximity_radius
    deltas = _DELTAS.to(dev)                    # [5, 2]

    if policy_aware:
        # Per-agent action probabilities: softmax of each agent's 5 logits at its cell.
        # logits is [1, 5, H, W]; positions[:, 0] is rows, [:, 1] is cols.
        agent_logits = logits[0, :, positions[:, 0], positions[:, 1]].T  # [N, 5]
        agent_probs  = torch.softmax(agent_logits, dim=-1)                # [N, 5]

    for i in range(N):
        ri, ci = positions[i, 0].item(), positions[i, 1].item()
        for a in range(5):
            target = pos_f[i] + deltas[a]                  # [2]
            diff = (target.unsqueeze(0) - pos_f).abs()     # [N, 2]
            linf = diff.max(dim=-1).values                 # [N]

            if policy_aware:
                # For each neighbour j, find which of j's 5 actions (if any) would
                # land j at target. That action's probability is w_j.
                # Action b moves j to pos_j + deltas[b]. We need pos_j + deltas[b] == target,
                # i.e. deltas[b] == target - pos_j.
                offsets = (target.unsqueeze(0) - pos_f).to(dev)        # [N, 2]
                # Match each row of offsets against deltas to find action index
                # deltas: [5, 2]; offsets: [N, 2]. Compare:
                # equal[n, b] = (offsets[n] == deltas[b]).all()
                eq = (offsets.unsqueeze(1) == deltas.unsqueeze(0)).all(dim=-1)  # [N, 5]
                # Probability j is at target = sum over matched actions of probs
                w = (eq.float() * agent_probs).sum(dim=-1)  # [N]
            else:
                w = ((R + 1 - linf) / (R + 1)).clamp(min=0.0)  # [N]

            w[i] = 0.0  # no self-penalty
            penalty = alpha * (w.to(dev) * phi).sum()
            shaped[0, a, ri, ci] -= penalty

    return shaped
