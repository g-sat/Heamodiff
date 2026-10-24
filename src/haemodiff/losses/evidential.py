"""Dirichlet classification and normal-inverse-gamma regression losses."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def dirichlet_loss(alpha: torch.Tensor, target: torch.Tensor, kl_weight: float = 0.05) -> torch.Tensor:
    """Sensoy evidential classification. ``target`` is a class index of shape ``(B,)``."""
    evidence_target = F.one_hot(target.long(), num_classes=alpha.shape[-1]).float()
    strength = alpha.sum(dim=-1, keepdim=True)
    probs = alpha / strength
    error = ((evidence_target - probs) ** 2).sum(dim=-1)
    variance = (probs * (1 - probs) / (strength + 1)).sum(dim=-1)
    bayes = error + variance
    tilde = evidence_target + (1 - evidence_target) * alpha
    kl = _dirichlet_kl(tilde)
    return (bayes + kl_weight * kl).mean()


def dirichlet_loss_per_sample(alpha: torch.Tensor, target: torch.Tensor, kl_weight: float = 0.05) -> torch.Tensor:
    evidence_target = F.one_hot(target.long(), num_classes=alpha.shape[-1]).float()
    strength = alpha.sum(dim=-1, keepdim=True)
    probs = alpha / strength
    error = ((evidence_target - probs) ** 2).sum(dim=-1)
    variance = (probs * (1 - probs) / (strength + 1)).sum(dim=-1)
    tilde = evidence_target + (1 - evidence_target) * alpha
    return error + variance + kl_weight * _dirichlet_kl(tilde)


def _dirichlet_kl(alpha: torch.Tensor) -> torch.Tensor:
    ones = torch.ones_like(alpha)
    sum_alpha = alpha.sum(dim=-1)
    sum_ones = ones.sum(dim=-1)
    t1 = torch.lgamma(sum_alpha) - torch.lgamma(sum_ones)
    t2 = (torch.lgamma(ones) - torch.lgamma(alpha)).sum(dim=-1)
    dig = torch.digamma(alpha) - torch.digamma(sum_alpha).unsqueeze(-1)
    t3 = ((alpha - ones) * dig).sum(dim=-1)
    return t1 + t2 + t3


def nig_nll(y: torch.Tensor, gamma: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    omega = 2 * beta * (1 + nu)
    nll = (
        0.5 * torch.log(y.new_tensor(math.pi) / nu)
        - alpha * torch.log(omega)
        + (alpha + 0.5) * torch.log(nu * (y - gamma) ** 2 + omega)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )
    return nll


def nig_regularizer(y: torch.Tensor, gamma: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return (y - gamma).abs() * (2 * nu + alpha)


def nig_loss(
    y: torch.Tensor,
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    reg_weight: float = 0.01,
) -> torch.Tensor:
    return (nig_nll(y, gamma, nu, alpha, beta) + reg_weight * nig_regularizer(y, gamma, nu, alpha)).mean()


def critic_loss(parts: dict[str, torch.Tensor], ef: torch.Tensor, severe: torch.Tensor, reduced: torch.Tensor) -> torch.Tensor:
    loss = dirichlet_loss(parts["severe_alpha"], severe)
    loss = loss + dirichlet_loss(parts["reduced_alpha"], reduced)
    scaled_ef = ef / 100.0
    loss = loss + nig_loss(scaled_ef, parts["gamma"], parts["nu"], parts["alpha"], parts["beta"])
    return loss


def critic_loss_per_sample(parts: dict[str, torch.Tensor], ef: torch.Tensor, severe: torch.Tensor, reduced: torch.Tensor) -> torch.Tensor:
    loss = dirichlet_loss_per_sample(parts["severe_alpha"], severe)
    loss = loss + dirichlet_loss_per_sample(parts["reduced_alpha"], reduced)
    scaled_ef = ef / 100.0
    loss = loss + nig_nll(scaled_ef, parts["gamma"], parts["nu"], parts["alpha"], parts["beta"])
    loss = loss + 0.01 * nig_regularizer(scaled_ef, parts["gamma"], parts["nu"], parts["alpha"])
    return loss
