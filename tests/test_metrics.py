"""Metric definitions used by the result tables."""

from __future__ import annotations

import numpy as np
import torch

from haemodiff.eval.metrics import auroc, binary_f1, expected_calibration_error, ms_ssim, relative_change, wasserstein


def test_perfect_scores_and_calibration():
    labels = np.array([0, 0, 1, 1])
    assert auroc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == 1
    assert binary_f1(labels, np.array([0.1, 0.2, 0.8, 0.9])) == 1
    ece, _ = expected_calibration_error(np.array([0.0, 0.0, 1.0, 1.0]), labels, n_bins=15)
    assert ece < 1e-6
    assert relative_change(0.5, 0.25) == 1
    assert wasserstein(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == 0


def test_identical_frames_have_high_ms_ssim():
    frame = torch.rand(2, 1, 64, 64)
    score = ms_ssim(frame, frame.clone())
    assert float(score.mean()) > 0.95
