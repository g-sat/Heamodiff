"""Seeded unpaired clinical covariates.

These fields are drawn from fixed marginals. They are not measurements from the
EchoNet patients and must keep the ``simulated_`` prefix.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    "simulated_factor_activity",
    "simulated_inhibitor_status",
    "simulated_annualized_bleed_rate",
    "simulated_age",
    "simulated_sex",
)

# Fixed scales so normalization does not depend on a patient cohort.
_SCALE = np.array([150.0, 1.0, 20.0, 100.0, 1.0], dtype=np.float32)


def sample_clinical(n: int, seed: int) -> np.ndarray:
    """Return an ``(n, 5)`` array. The draw depends only on ``n`` and ``seed``."""
    if n < 0:
        raise ValueError("n must be non-negative")
    rng = np.random.default_rng(seed)
    factor = np.clip(rng.lognormal(mean=3.0, sigma=0.6, size=n), 0.0, 150.0)
    inhibitor = rng.random(n) < 0.15
    bleed = rng.gamma(shape=2.0, scale=2.5, size=n)
    age = rng.uniform(18.0, 80.0, size=n)
    sex = rng.integers(0, 2, size=n).astype(np.float64)
    raw = np.column_stack([factor, inhibitor.astype(np.float64), bleed, age, sex])
    return raw.astype(np.float32)


def normalize_clinical(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float32)
    scaled = values / _SCALE
    return np.clip(scaled, 0.0, 2.0).astype(np.float32)


def assert_simulated_names() -> None:
    for name in FEATURE_NAMES:
        if not name.startswith("simulated_"):
            raise AssertionError(name)
