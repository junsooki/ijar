"""Utility to extract agent positions from a RAILGUN feature tensor.

Feature channel 1: agent ID at each cell (1-indexed; 0 = empty).
"""

import torch


def extract_agent_positions(feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Read agent positions from feature channel 1.

    Parameters
    ----------
    feat : Tensor [C, H, W]
        RAILGUN feature tensor on any device.

    Returns
    -------
    positions : Tensor [N, 2] int64  — (row, col) for each agent
    agent_ids : Tensor [N]   int64  — 1-indexed agent IDs, sorted ascending
    """
    ch1 = feat[1]                                          # [H, W]
    nonzero = (ch1 > 0).nonzero(as_tuple=False)           # [N, 2]
    if nonzero.size(0) == 0:
        empty = torch.zeros(0, dtype=torch.long, device=feat.device)
        return torch.zeros(0, 2, dtype=torch.long, device=feat.device), empty
    agent_ids = ch1[nonzero[:, 0], nonzero[:, 1]].long()  # [N]
    order     = agent_ids.argsort()
    return nonzero[order].cpu(), agent_ids[order].cpu()
