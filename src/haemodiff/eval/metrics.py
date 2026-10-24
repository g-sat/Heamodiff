"""Metric implementations used by the four evaluation pillars."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores).astype(np.float64)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks for ties.
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end > start + 1:
            average = 0.5 * (start + 1 + end)
            ranks[order[start:end]] = average
        start = end
    sum_positive = ranks[labels == 1].sum()
    return float((sum_positive - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))


def auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores).astype(np.float64)
    positives = int((labels == 1).sum())
    if positives == 0 or len(labels) == 0:
        return float("nan")
    order = np.argsort(-scores)
    y = labels[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def binary_f1(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> float:
    labels = np.asarray(labels).astype(np.int64)
    predicted = (np.asarray(scores) >= threshold).astype(np.int64)
    tp = int(((predicted == 1) & (labels == 1)).sum())
    fp = int(((predicted == 1) & (labels == 0)).sum())
    fn = int(((predicted == 0) & (labels == 1)).sum())
    denom = 2 * tp + fp + fn
    if denom == 0:
        return float("nan")
    return float(2 * tp / denom)


def relative_change(value: float, baseline: float) -> float:
    if baseline == 0 or math.isnan(baseline) or math.isnan(value):
        return float("nan")
    return float((value - baseline) / baseline)


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> tuple[float, list[dict]]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64)
    predicted = (probabilities >= 0.5).astype(np.int64)
    confidence = np.where(predicted == 1, probabilities, 1.0 - probabilities)
    correct = (predicted == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(probabilities)
    ece = 0.0
    bins = []
    for index in range(n_bins):
        lo, hi = edges[index], edges[index + 1]
        if index == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        count = int(mask.sum())
        acc = float(correct[mask].mean()) if count else float("nan")
        conf = float(confidence[mask].mean()) if count else float("nan")
        if count:
            ece += (count / total) * abs(acc - conf)
        bins.append({"lo": float(lo), "hi": float(hi), "count": count, "accuracy": acc, "confidence": conf})
    return float(ece), bins


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    coefficient = spearmanr(x, y).statistic
    return float(coefficient)


def wasserstein(real: np.ndarray, generated: np.ndarray) -> float:
    from scipy.stats import wasserstein_distance

    return float(wasserstein_distance(np.asarray(real, dtype=np.float64), np.asarray(generated, dtype=np.float64)))


def ks_two_sample(real: np.ndarray, generated: np.ndarray) -> tuple[float, float]:
    from scipy.stats import ks_2samp

    result = ks_2samp(np.asarray(real, dtype=np.float64), np.asarray(generated, dtype=np.float64))
    return float(result.statistic), float(result.pvalue)


def _gaussian_window(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    window = kernel[:, None] * kernel[None, :]
    return window.view(1, 1, size, size)


def _ssim_components(x: torch.Tensor, y: torch.Tensor, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu_x = F.conv2d(x, window, padding=0)
    mu_y = F.conv2d(y, window, padding=0)
    sigma_x = F.conv2d(x * x, window, padding=0) - mu_x**2
    sigma_y = F.conv2d(y * y, window, padding=0) - mu_y**2
    sigma_xy = F.conv2d(x * y, window, padding=0) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    luminance = (2 * mu_x * mu_y + c1) / (mu_x**2 + mu_y**2 + c1)
    contrast = (2 * sigma_xy + c2) / (sigma_x + sigma_y + c2)
    return luminance.mean(dim=(1, 2, 3)), contrast.clamp_min(0).mean(dim=(1, 2, 3))


def ms_ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Five-scale SSIM. ``x`` and ``y`` are ``(N, 1, H, W)`` in ``[0, 1]``."""
    weights = x.new_tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    window = _gaussian_window(11, 1.5, x.device, x.dtype)
    levels_contrast = []
    level_luminance = None
    current_x, current_y = x, y
    for _level in range(5):
        if min(current_x.shape[-2:]) < 11:
            break
        luminance, contrast = _ssim_components(current_x, current_y, window)
        levels_contrast.append(contrast)
        level_luminance = luminance
        current_x = F.avg_pool2d(current_x, 2)
        current_y = F.avg_pool2d(current_y, 2)
    if not levels_contrast or level_luminance is None:
        return torch.ones(x.shape[0], device=x.device)
    used = weights[: len(levels_contrast)]
    used = used / used.sum()
    stacked = torch.stack(levels_contrast[:-1] + [levels_contrast[-1] * level_luminance], dim=0).clamp_min(1e-6)
    return torch.exp(torch.sum(used.view(-1, 1) * torch.log(stacked), dim=0))


def temporal_ms_ssim(videos: torch.Tensor) -> float:
    """Mean MS-SSIM between consecutive frames. ``videos`` is ``(N, T, 1, H, W)``."""
    scores = []
    for time in range(videos.shape[1] - 1):
        scores.append(ms_ssim(videos[:, time], videos[:, time + 1]))
    return float(torch.cat(scores).mean())


def laplacian_variance(videos: torch.Tensor) -> np.ndarray:
    """Sharpness of each clip, averaged over frames. ``videos`` is ``(N, T, 1, H, W)``."""
    kernel = videos.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3)
    n, t, _, h, w = videos.shape
    flat = videos.reshape(n * t, 1, h, w)
    response = F.conv2d(flat, kernel, padding=1)
    per_frame = response.var(dim=(1, 2, 3)).reshape(n, t).mean(dim=1)
    return per_frame.detach().cpu().numpy()


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    from scipy.linalg import sqrtm

    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


def frame_fid(real: torch.Tensor, generated: torch.Tensor, device: torch.device) -> float | None:
    """Inception FID on middle frames. Returns None when torchvision weights are unavailable."""
    try:
        from torchvision.models import Inception_V3_Weights, inception_v3
    except Exception:
        return None
    try:
        net = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    except Exception:
        return None
    net.fc = torch.nn.Identity()
    net.eval().to(device)

    def embed(frames: torch.Tensor) -> np.ndarray:
        batch = frames
        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)
        batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
        features = []
        for start in range(0, len(batch), 16):
            with torch.no_grad():
                features.append(net(batch[start : start + 16].to(device)).detach().cpu().numpy())
        return np.concatenate(features, axis=0)

    def stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return values.mean(axis=0), np.cov(values, rowvar=False)

    real_stats = stats(embed(real))
    fake_stats = stats(embed(generated))
    return frechet_distance(real_stats[0], real_stats[1], fake_stats[0], fake_stats[1])
