"""Loss-threshold and shadow-model membership inference."""

from __future__ import annotations

import numpy as np


def loss_threshold_auc(member_losses: np.ndarray, nonmember_losses: np.ndarray) -> float:
    """Lower loss is treated as evidence of membership."""
    from haemodiff.eval.metrics import auroc

    member_losses = np.asarray(member_losses, dtype=np.float64)
    nonmember_losses = np.asarray(nonmember_losses, dtype=np.float64)
    labels = np.concatenate(
        [np.ones(len(member_losses), dtype=np.int64), np.zeros(len(nonmember_losses), dtype=np.int64)]
    )
    scores = np.concatenate([-member_losses, -nonmember_losses])
    return auroc(labels, scores)


def _fit_logistic(losses: np.ndarray, labels: np.ndarray, steps: int = 400, lr: float = 0.1) -> tuple[float, float]:
    mean = float(losses.mean())
    std = float(losses.std()) + 1e-6
    x = (losses - mean) / std
    weight = 0.0
    bias = 0.0
    y = labels.astype(np.float64)
    for _ in range(steps):
        pred = 1 / (1 + np.exp(-(weight * x + bias)))
        grad = pred - y
        weight -= lr * float(np.mean(grad * x))
        bias -= lr * float(np.mean(grad))
    return weight, bias, mean, std


def _predict(losses: np.ndarray, weight: float, bias: float, mean: float, std: float) -> np.ndarray:
    x = (losses - mean) / std
    return 1 / (1 + np.exp(-(weight * x + bias)))


def shadow_attack_advantage(
    shadow_member_losses: np.ndarray,
    shadow_nonmember_losses: np.ndarray,
    target_member_losses: np.ndarray,
    target_nonmember_losses: np.ndarray,
) -> float:
    """Train a one-feature attack on the shadow model and return target AUC minus 0.5."""
    from haemodiff.eval.metrics import auroc

    shadow_losses = np.concatenate([np.asarray(shadow_member_losses), np.asarray(shadow_nonmember_losses)])
    shadow_labels = np.concatenate(
        [
            np.ones(len(shadow_member_losses)),
            np.zeros(len(shadow_nonmember_losses)),
        ]
    )
    weight, bias, mean, std = _fit_logistic(shadow_losses.astype(np.float64), shadow_labels)
    target_losses = np.concatenate([np.asarray(target_member_losses), np.asarray(target_nonmember_losses)])
    target_labels = np.concatenate(
        [
            np.ones(len(target_member_losses), dtype=np.int64),
            np.zeros(len(target_nonmember_losses), dtype=np.int64),
        ]
    )
    scores = _predict(target_losses.astype(np.float64), weight, bias, mean, std)
    return float(auroc(target_labels, scores) - 0.5)
