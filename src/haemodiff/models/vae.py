"""Per-frame variational autoencoder. Video time is handled by the DiT."""

from __future__ import annotations

import torch
from torch import nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(8, out_ch),
        nn.SiLU(),
    )


def _up_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(8, out_ch),
        nn.SiLU(),
    )


class FrameVAE(nn.Module):
    def __init__(self, latent_channels: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.encoder = nn.Sequential(
            _conv_block(1, c),
            _conv_block(c, c * 2),
            _conv_block(c * 2, c * 4),
        )
        self.to_stats = nn.Conv2d(c * 4, latent_channels * 2, kernel_size=1)
        self.from_latent = nn.Conv2d(latent_channels, c * 4, kernel_size=1)
        self.decoder = nn.Sequential(
            _up_block(c * 4, c * 2),
            _up_block(c * 2, c),
            nn.ConvTranspose2d(c, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self.latent_channels = latent_channels

    def encode(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stats = self.to_stats(self.encoder(frames))
        mu, logvar = stats.chunk(2, dim=1)
        return mu, logvar.clamp(-20, 20)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.from_latent(latent))

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(frames)
        latent = self.reparameterize(mu, logvar)
        return self.decode(latent), mu, logvar

    def encode_video(self, video: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """``video`` is ``(B, T, 1, H, W)``. Returns ``(B, T, C, h, w)``."""
        b, t, _, _, _ = video.shape
        mu, logvar = self.encode(video.reshape(b * t, 1, video.shape[-2], video.shape[-1]))
        latent = self.reparameterize(mu, logvar) if sample else mu
        return latent.reshape(b, t, self.latent_channels, latent.shape[-2], latent.shape[-1])

    def decode_video(self, latent: torch.Tensor) -> torch.Tensor:
        b, t, _, h, w = latent.shape
        frames = self.decode(latent.reshape(b * t, self.latent_channels, h, w))
        return frames.reshape(b, t, 1, frames.shape[-2], frames.shape[-1])


def vae_loss(recon: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, kl_weight: float) -> tuple[torch.Tensor, dict]:
    rec = torch.nn.functional.mse_loss(recon, target)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = rec + kl_weight * kl
    return total, {"recon": rec.detach(), "kl": kl.detach()}
