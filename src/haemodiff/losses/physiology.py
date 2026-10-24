"""EMA-balanced physiological loss with a warmup ramp."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Dice on frames whose valid-mask flag is 1. ``logits`` and ``target`` are (B, T, 1, H, W)."""
    pred = torch.sigmoid(logits)
    weights = valid.view(valid.shape[0], valid.shape[1], 1, 1, 1)
    if float(weights.sum()) == 0:
        return logits.new_zeros(())
    intersection = (pred * target * weights).sum(dim=(2, 3, 4))
    denom = ((pred + target) * weights).sum(dim=(2, 3, 4))
    dice = (2 * intersection + 1) / (denom + 1)
    frame_ok = valid > 0
    if int(frame_ok.sum()) == 0:
        return logits.new_zeros(())
    return 1 - dice[frame_ok].mean()


def temporal_total_variation(video: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    """Mean absolute frame difference. Padded steps are ignored."""
    diff = (video[:, 1:] - video[:, :-1]).abs().mean(dim=(2, 3, 4))
    keep = (~pad[:, 1:]) & (~pad[:, :-1])
    if int(keep.sum()) == 0:
        return video.new_zeros(())
    return diff[keep].mean()


def physiology_terms(
    video: torch.Tensor,
    mask_logits: torch.Tensor,
    ef_pred: torch.Tensor,
    ef_target: torch.Tensor,
    mask_target: torch.Tensor,
    valid_mask: torch.Tensor,
    pad: torch.Tensor,
    use_ef: bool,
    use_mask: bool,
) -> dict[str, torch.Tensor]:
    ef_loss = (ef_pred - ef_target).abs().mean() if use_ef else video.new_zeros(())
    dice_loss = soft_dice_loss(mask_logits, mask_target, valid_mask) if use_mask else video.new_zeros(())
    tv_loss = temporal_total_variation(video, pad)
    return {"ef": ef_loss, "dice": dice_loss, "tv": tv_loss}


class PhysiologyBalancer:
    """Scale EF, Dice, and temporal terms so none of them drowns the diffusion loss."""

    def __init__(
        self,
        warmup_steps: int = 1000,
        ramp_steps: int = 1000,
        max_weight: float = 0.1,
        patience: int = 50,
        ema_decay: float = 0.99,
        shares: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    ) -> None:
        self.warmup_steps = int(warmup_steps)
        self.ramp_steps = int(ramp_steps)
        self.max_weight = float(max_weight)
        self.patience = int(patience)
        self.ema_decay = float(ema_decay)
        self.shares = shares
        self.step = 0
        self.ema: dict[str, float] | None = None
        self.over_count = 0
        self.frozen = False
        self.frozen_fraction = 0.0
        self.last_safe_fraction = 0.0
        self.last_contributions: dict[str, float] = {}
        self.last_fraction = 0.0

    def scheduled_fraction(self) -> float:
        if self.frozen:
            return self.frozen_fraction
        if self.step < self.warmup_steps:
            return 0.0
        if self.ramp_steps <= 0:
            return self.max_weight
        progress = min(1.0, (self.step - self.warmup_steps) / self.ramp_steps)
        return progress * self.max_weight

    def __call__(
        self,
        diffusion_loss: torch.Tensor,
        terms: dict[str, torch.Tensor],
        active_keys: tuple[str, ...] = ("ef", "dice", "tv"),
    ) -> torch.Tensor:
        keys = [key for key in ("ef", "dice", "tv") if key in active_keys]
        values = torch.stack([terms[key] for key in keys])
        detached = [float(terms[key].detach()) for key in keys]
        if self.ema is None:
            self.ema = {key: max(value, 1e-6) for key, value in zip(keys, detached)}
        else:
            for key, value in zip(keys, detached):
                previous = self.ema.get(key, max(value, 1e-6))
                self.ema[key] = self.ema_decay * previous + (1 - self.ema_decay) * max(value, 1e-6)
        share = 1.0 / len(keys)
        weighted = []
        contributions = {}
        for key in keys:
            scale = share / (self.ema[key] + 1e-6)
            piece = terms[key] * scale
            weighted.append(piece)
            contributions[key] = float(piece.detach())
        balanced = torch.stack(weighted).sum()
        fraction = self.scheduled_fraction()
        proposed = balanced * (
            fraction * diffusion_loss.detach() / (balanced.detach().clamp_min(1e-8))
        )
        applied_mag = float(proposed.detach())
        diff_mag = float(diffusion_loss.detach())
        if (not self.frozen) and self.step >= self.warmup_steps and applied_mag > diff_mag > 0:
            self.over_count += 1
            if self.over_count >= self.patience:
                self.frozen = True
                self.frozen_fraction = self.last_safe_fraction
                fraction = self.frozen_fraction
                proposed = balanced * (
                    fraction * diffusion_loss.detach() / (balanced.detach().clamp_min(1e-8))
                )
        else:
            self.over_count = 0
            if not self.frozen:
                self.last_safe_fraction = fraction
        self.last_contributions = contributions
        self.last_fraction = fraction
        self.step += 1
        return proposed

    def state_dict(self) -> dict:
        return {
            "step": self.step,
            "ema": self.ema,
            "over_count": self.over_count,
            "frozen": self.frozen,
            "frozen_fraction": self.frozen_fraction,
            "last_safe_fraction": self.last_safe_fraction,
        }

    def load_state_dict(self, state: dict) -> None:
        self.step = int(state["step"])
        self.ema = state["ema"]
        self.over_count = int(state["over_count"])
        self.frozen = bool(state["frozen"])
        self.frozen_fraction = float(state["frozen_fraction"])
        self.last_safe_fraction = float(state["last_safe_fraction"])


def masked_bce(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = valid.view(valid.shape[0], valid.shape[1], 1, 1, 1)
    if float(weights.sum()) == 0:
        return logits.new_zeros(())
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)
