"""Frozen ejection-fraction and ventricle-mask network trained on real clips."""

from __future__ import annotations

import torch
from torch import nn


class EFNet(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.trunk = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 4, stride=2, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, c * 2),
            nn.SiLU(),
            nn.Conv2d(c * 2, c * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, c * 2),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.ef_head = nn.Linear(c * 2, 1)
        self.mask_head = nn.Sequential(
            nn.Conv2d(c * 2, c, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False),
            nn.Conv2d(c, 1, 1),
        )

    def forward(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``video`` is ``(B, T, 1, H, W)`` in ``[0, 1]``.

        Returns EF ``(B,)`` and mask logits ``(B, T, 1, H, W)``.
        """
        b, t, _, h, w = video.shape
        flat = video.reshape(b * t, 1, h, w)
        feat = self.trunk(flat)
        pooled = self.pool(feat).reshape(b, t, -1).mean(dim=1)
        ef = self.ef_head(pooled).squeeze(-1)
        mask_small = self.mask_head[0](feat)
        mask_small = self.mask_head[1](mask_small)
        mask = self.mask_head[2](mask_small)
        mask = self.mask_head[3](mask)
        if mask.shape[-2:] != (h, w):
            mask = torch.nn.functional.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
        return ef, mask.reshape(b, t, 1, h, w)
