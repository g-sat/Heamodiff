"""Physiology balancer and a small diffusion/EF forward."""

from __future__ import annotations

import torch

from haemodiff.losses.physiology import PhysiologyBalancer, physiology_terms
from haemodiff.models.dit import VideoDiT
from haemodiff.models.ef_net import EFNet
from haemodiff.models.vae import FrameVAE


class ConstantEF(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, video: torch.Tensor):
        batch = video.shape[0]
        ef = video.new_full((batch,), self.value)
        logits = torch.zeros(batch, video.shape[1], 1, video.shape[-2], video.shape[-1], device=video.device)
        return ef, logits


def test_matching_ef_has_lower_physiology_loss():
    video = torch.rand(2, 4, 1, 16, 16)
    mask = torch.zeros(2, 4, 1, 16, 16)
    valid = torch.zeros(2, 4)
    valid[:, 0] = 1
    pad = torch.zeros(2, 4, dtype=torch.bool)
    target = torch.tensor([40.0, 40.0])
    matched_ef, matched_mask = ConstantEF(40.0)(video)
    missed_ef, missed_mask = ConstantEF(10.0)(video)
    matched = physiology_terms(video, matched_mask, matched_ef, target, mask, valid, pad, True, True)
    missed = physiology_terms(video, missed_mask, missed_ef, target, mask, valid, pad, True, True)
    assert float(matched["ef"]) < float(missed["ef"])


def test_warmup_is_zero_and_ema_shrinks_a_dominant_term():
    balancer = PhysiologyBalancer(warmup_steps=3, ramp_steps=1, max_weight=0.1, patience=100)
    diffusion = torch.tensor(2.0)
    terms = {"ef": torch.tensor(1000.0), "dice": torch.tensor(1.0), "tv": torch.tensor(1.0)}
    applied = balancer(diffusion, terms)
    assert float(applied) == 0.0
    hot = PhysiologyBalancer(warmup_steps=0, ramp_steps=0, max_weight=1.0, patience=100, ema_decay=0.0)
    hot(diffusion, terms)
    assert hot.last_contributions["ef"] < 10 * hot.last_contributions["dice"]


def test_tiny_vae_and_dit_backward():
    vae = FrameVAE(latent_channels=4, base_channels=8)
    frames = torch.rand(2, 1, 32, 32)
    recon, mu, _ = vae(frames)
    assert recon.shape == frames.shape
    assert mu.shape[-1] == 4
    dit = VideoDiT(latent_channels=4, hidden=32, depth=2, heads=4, patch=2, grad_checkpoint=False)
    latent = torch.randn(1, 4, 4, 8, 8)
    noise = dit(
        latent,
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, 5),
        torch.zeros(1),
        torch.zeros(1, 4, 1, 8, 8),
        torch.zeros(1, 4),
    )
    assert noise.shape == latent.shape
    noise.mean().backward()
    ef_net = EFNet(base_channels=8)
    ef, masks = ef_net(torch.rand(1, 4, 1, 32, 32))
    assert ef.shape == (1,)
    assert masks.shape[1] == 4
