"""Evidential critic. A single tensor output keeps Opacus grad sampling valid."""

from __future__ import annotations

import torch
from torch import nn


class EvidentialCritic(nn.Module):
    """GroupNorm CNN. Time is the channel axis so every op is Opacus-friendly.

    Output layout per row: severe Dirichlet evidence (2), reduced Dirichlet
    evidence (2), NIG raw parameters (4).
    """

    def __init__(self, in_channels: int = 16, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1),
            nn.GroupNorm(8, c),
            nn.ReLU(),
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, c * 2),
            nn.ReLU(),
            nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, c * 4),
            nn.ReLU(),
            nn.Conv2d(c * 4, c * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, c * 4),
            nn.ReLU(),
        )
        self.head = nn.Linear(c * 4, 8)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim == 5:
            video = video.squeeze(2)
        features = self.features(video).mean(dim=(-2, -1))
        return self.head(features)


def split_critic_output(raw: torch.Tensor) -> dict[str, torch.Tensor]:
    severe = torch.nn.functional.softplus(raw[:, 0:2]) + 1.0
    reduced = torch.nn.functional.softplus(raw[:, 2:4]) + 1.0
    gamma = raw[:, 4]
    nu = torch.nn.functional.softplus(raw[:, 5]) + 1e-3
    alpha = torch.nn.functional.softplus(raw[:, 6]) + 1.0 + 1e-3
    beta = torch.nn.functional.softplus(raw[:, 7]) + 1e-3
    return {
        "severe_alpha": severe,
        "reduced_alpha": reduced,
        "gamma": gamma,
        "nu": nu,
        "alpha": alpha,
        "beta": beta,
    }


def vacuity(alpha: torch.Tensor) -> torch.Tensor:
    return alpha.shape[-1] / alpha.sum(dim=-1).clamp_min(1e-6)


def class_probability(alpha: torch.Tensor) -> torch.Tensor:
    return alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def aleatoric_variance(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return beta / (alpha - 1).clamp_min(1e-3)
