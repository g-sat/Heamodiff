"""DDPM schedule and x0/noise conversions."""

from __future__ import annotations

import torch


class DiffusionSchedule:
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02) -> None:
        beta = torch.linspace(beta_start, beta_end, timesteps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.timesteps = timesteps
        self.beta = beta
        self.alpha_bar = alpha_bar

    def to(self, device: torch.device) -> "DiffusionSchedule":
        self.beta = self.beta.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        return self

    def q_sample(self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        bar = self._extract(self.alpha_bar, timesteps, x0.shape)
        return bar.sqrt() * x0 + (1.0 - bar).sqrt() * noise

    def predict_x0(self, xt: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        bar = self._extract(self.alpha_bar, timesteps, xt.shape)
        return (xt - (1.0 - bar).sqrt() * noise) / bar.sqrt().clamp_min(1e-8)

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        out = values.gather(0, timesteps.long())
        return out.view(-1, *([1] * (len(shape) - 1)))


def diffusion_mse(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((predicted - target) ** 2)
