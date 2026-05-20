"""Iterative joint-action refinement policy (task #48).

Wraps RAILGUN UNet with a feedback encoder that processes previous-action +
conflict features. The combined representation produces refined per-cell logits.

Architecture:
  X        [B, 6, H, W]   ─► UNet body ─► penult [B, C, H, W] ─┐
                                                                ├─► sum ─► output_conv ─► logits [B, 5, H, W]
  (A, C)   [B, 16, H, W]  ─► feedback encoder ────► [B, C, H, W]┘

At init: feedback encoder is zero-init at its final layer, so its contribution
is 0 → policy ≡ RAILGUN. This gives a clean warm-start.

Iterative inference:
  A^0 = 0, C^0 = 0
  for r = 1..K:
    logits^r = policy(X, A^{r-1}, C^{r-1})
    A^r      = argmax(logits^r) at agent cells (5-channel one-hot)
    C^r      = compute_conflict_features(A^r, ...)
    if no_conflict(C^r): break
  return logits^K
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedbackEncoder(nn.Module):
    """Small conv stack that processes (action map, conflict map) into a feature
    representation in the SAME shape as UNet's penultimate features.

    Input:  [B, 16, H, W]  (5-channel action one-hot + 11-channel conflict map)
    Output: [B, out_channels, H, W]

    Last conv is zero-init so that at t=0 (zero feedback) the encoder output is
    exactly zero, preserving the underlying RAILGUN policy's behavior.
    """
    def __init__(self, in_channels: int = 16, out_channels: int = 64, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1),
        )
        # Zero-init the final conv so at first pass (zero feedback) we contribute
        # exactly 0 and the policy reduces to vanilla RAILGUN.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IterativeRefinementPolicy(nn.Module):
    """RAILGUN UNet + feedback encoder for iterative joint-action refinement.

    Forward signature is the standard MAPF policy contract — input [B, 6, H, W]
    and returns (logits [B, 5, H, W], None) — but ALSO accepts optional feedback
    `prev_action_map` and `prev_conflict_map` for non-first refinement passes.

    Usage in rollout:
        # First pass:
        logits, _ = policy(feat)
        # Subsequent passes:
        logits, _ = policy(feat, prev_action_map=A_prev, prev_conflict_map=C_prev)
    """

    def __init__(self, unet: nn.Module, hidden: int = 32):
        super().__init__()
        self.unet = unet
        # Infer UNet penultimate channel count from up4's last Conv2d.
        feature_channels = self._infer_feature_channels(unet)
        # Feedback encoder produces features in the SAME shape as UNet penultimate.
        self.feedback_encoder = FeedbackEncoder(
            in_channels=16,           # 5 action + 11 conflict
            out_channels=feature_channels,
            hidden=hidden,
        )
        self.n_classes = unet.n_classes

    @staticmethod
    def _infer_feature_channels(unet) -> int:
        for m in reversed(list(unet.up4.modules())):
            if isinstance(m, nn.Conv2d):
                return m.out_channels
        raise RuntimeError("Could not infer up4 output channels")

    def _unet_penultimate(self, x: torch.Tensor) -> torch.Tensor:
        u = self.unet
        x1 = u.input_conv(x)
        x2 = u.down1(x1)
        x3 = u.down2(x2)
        x4 = u.down3(x3)
        x5 = u.down4(x4)
        h = u.up1(x5, x4)
        h = u.up2(h, x3)
        h = u.up3(h, x2)
        h = u.up4(h, x1)
        return h

    def forward(
        self,
        feat: torch.Tensor,
        prev_action_map: "torch.Tensor | None" = None,
        prev_conflict_map: "torch.Tensor | None" = None,
    ):
        """Run one refinement pass.

        feat              : [B, 6, H, W]
        prev_action_map   : [B, 5, H, W]  (zeros for first pass)
        prev_conflict_map : [B, 11, H, W] (zeros for first pass)

        Returns (logits [B, 5, H, W], None) to match the existing UNet contract.
        """
        B, _, H, W = feat.shape
        penult = self._unet_penultimate(feat)  # [B, C, H, W]

        if prev_action_map is None:
            prev_action_map = torch.zeros(B, 5, H, W, device=feat.device, dtype=feat.dtype)
        if prev_conflict_map is None:
            prev_conflict_map = torch.zeros(B, 11, H, W, device=feat.device, dtype=feat.dtype)

        feedback = torch.cat([prev_action_map, prev_conflict_map], dim=1)  # [B, 16, H, W]
        fb_feat = self.feedback_encoder(feedback)                          # [B, C, H, W]

        combined = penult + fb_feat                                        # additive
        logits = self.unet.output_conv(combined)                           # [B, 5, H, W]
        return logits, None


def _smoke_test():
    """Quick test: load RAILGUN, wrap, verify init matches RAILGUN bitwise."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "RAILGUN"))
    from models.unet import UNet

    u = UNet(n_channels=6, n_classes=5, first_layer_channels=64,
             bilinear=False, blocks_per_stage=0)
    sd = torch.load(
        os.path.join(os.path.dirname(os.path.dirname(__file__)),
                     "results/checkpoints/railgun_pretrained.pt"),
        map_location="cpu", weights_only=True,
    )
    u.load_state_dict(sd if "unet" not in sd else sd["unet"], strict=False)
    u.eval()

    policy = IterativeRefinementPolicy(u).eval()

    feat = torch.zeros(1, 6, 16, 16)
    feat[0, 1, 5, 5] = 1
    feat[0, 1, 8, 8] = 2
    feat[0, 2, 12, 12] = 1
    feat[0, 2, 2, 2] = 2

    with torch.no_grad():
        r_logits, _ = u(feat)
        # With zero feedback, refinement output should equal RAILGUN's output.
        p_logits, _ = policy(feat)
    max_diff = (r_logits - p_logits).abs().max().item()
    print(f"output shape: {tuple(p_logits.shape)}")
    print(f"warm-start init: max logit diff vs RAILGUN = {max_diff:.6f} (expect ~0)")
    n_total = sum(p.numel() for p in policy.parameters())
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_feedback = sum(p.numel() for p in policy.feedback_encoder.parameters())
    print(f"params: total={n_total:,}  trainable={n_train:,}  feedback_encoder={n_feedback:,}")


if __name__ == "__main__":
    _smoke_test()
