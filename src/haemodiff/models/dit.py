"""Conditional video latent diffusion transformer with AdaLN-Zero blocks."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def _sinusoidal(length: int, dim: int, device: torch.device) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(-math.log(10000) * torch.arange(half, device=device, dtype=torch.float32) / max(half, 1))
    args = torch.arange(length, device=device, dtype=torch.float32)[:, None] * freq[None]
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    if dim % 2:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half)
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
        inner = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, inner),
            nn.GELU(),
            nn.Linear(inner, hidden),
        )
        self.ada = nn.Linear(hidden, 6 * hidden)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada(cond).chunk(6, dim=-1)
        normed = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class VideoDiT(nn.Module):
    def __init__(
        self,
        latent_channels: int = 4,
        hidden: int = 384,
        depth: int = 12,
        heads: int = 6,
        patch: int = 2,
        mlp_ratio: float = 4.0,
        clinical_dim: int = 5,
        grad_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.patch = patch
        self.hidden = hidden
        self.latent_channels = latent_channels
        in_ch = latent_channels + latent_channels + 1
        self.patch_embed = nn.Conv2d(in_ch, hidden, kernel_size=patch, stride=patch)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(clinical_dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.blocks = nn.ModuleList(DiTBlock(hidden, heads, mlp_ratio) for _ in range(depth))
        self.final_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.final_ada = nn.Linear(hidden, 2 * hidden)
        nn.init.zeros_(self.final_ada.weight)
        nn.init.zeros_(self.final_ada.bias)
        self.unpatch = nn.Linear(hidden, patch * patch * latent_channels)
        nn.init.zeros_(self.unpatch.weight)
        nn.init.zeros_(self.unpatch.bias)
        self.grad_checkpoint = grad_checkpoint

    def _positions(self, frames: int, tokens: int, device: torch.device) -> torch.Tensor:
        spatial = _sinusoidal(tokens, self.hidden, device).view(1, 1, tokens, self.hidden)
        temporal = _sinusoidal(frames, self.hidden, device).view(1, frames, 1, self.hidden)
        return (spatial + temporal).reshape(1, frames * tokens, self.hidden)

    def forward(
        self,
        latent: torch.Tensor,
        timesteps: torch.Tensor,
        clinical: torch.Tensor,
        ef: torch.Tensor,
        mask: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise. ``latent`` is ``(B, T, C, h, w)``."""
        b, t, _, h, w = latent.shape
        mask_lat = self._downsample_mask(mask, h, w)
        valid_map = valid.view(b, t, 1, 1, 1).expand(b, t, 1, h, w)
        tokens_in = torch.cat([latent, mask_lat, valid_map], dim=2)
        patched = self.patch_embed(tokens_in.reshape(b * t, tokens_in.shape[2], h, w))
        _, hidden, gh, gw = patched.shape
        seq = patched.reshape(b, t, hidden, gh * gw).permute(0, 1, 3, 2)
        seq = seq.reshape(b, t * gh * gw, hidden)
        seq = seq + self._positions(t, gh * gw, latent.device)
        cond = self.time_mlp(timestep_embedding(timesteps, self.hidden))
        cond = cond + self.cond_mlp(torch.cat([clinical, ef.view(b, 1)], dim=-1))
        for block in self.blocks:
            if self.grad_checkpoint and self.training:
                seq = checkpoint(block, seq, cond, use_reentrant=False)
            else:
                seq = block(seq, cond)
        shift, scale = self.final_ada(cond).chunk(2, dim=-1)
        seq = modulate(self.final_norm(seq), shift, scale)
        pixels = self.unpatch(seq)
        pixels = pixels.reshape(b, t, gh, gw, self.patch, self.patch, self.latent_channels)
        pixels = pixels.permute(0, 1, 6, 2, 4, 3, 5)
        return pixels.reshape(b, t, self.latent_channels, gh * self.patch, gw * self.patch)

    def _downsample_mask(self, mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
        b, t, _, h, w = mask.shape
        pooled = torch.nn.functional.adaptive_avg_pool2d(mask.reshape(b * t, 1, h, w), (height, width))
        return pooled.reshape(b, t, 1, height, width).repeat(1, 1, self.latent_channels, 1, 1)


def apply_condition_mode(
    clinical: torch.Tensor,
    ef: torch.Tensor,
    mask: torch.Tensor,
    valid: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if mode == "unconditional":
        return torch.zeros_like(clinical), torch.zeros_like(ef), torch.zeros_like(mask), torch.zeros_like(valid)
    if mode == "clinical":
        return clinical, ef, torch.zeros_like(mask), torch.zeros_like(valid)
    if mode == "mask":
        return torch.zeros_like(clinical), torch.zeros_like(ef), mask, valid
    if mode == "full":
        return clinical, ef, mask, valid
    raise ValueError(mode)
