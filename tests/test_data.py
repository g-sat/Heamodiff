"""Developer checks for covariates, tracings, and index rules. These are not the Kaggle run."""

from __future__ import annotations

import numpy as np

from haemodiff.data.audit import classify_case
from haemodiff.data.clinical import FEATURE_NAMES, sample_clinical
from haemodiff.data.dataset import assert_no_test_leak
from haemodiff.data.tracings import plan_window, polygon_from_segments, rasterize_polygon


def test_clinical_is_deterministic_and_prefixed():
    first = sample_clinical(8, seed=17)
    second = sample_clinical(8, seed=17)
    assert first.shape == (8, 5)
    assert np.allclose(first, second)
    assert list(FEATURE_NAMES) == [name for name in FEATURE_NAMES if name.startswith("simulated_")]
    assert not np.allclose(sample_clinical(8, seed=18), first)


def test_raster_square_and_degenerate():
    segments = np.array(
        [
            [10, 10, 40, 10],
            [10, 20, 40, 20],
            [10, 30, 40, 30],
            [10, 40, 40, 40],
        ],
        dtype=np.float64,
    )
    mask, ok = rasterize_polygon(polygon_from_segments(segments), 64, 64)
    assert ok
    assert mask[25, 25] == 1
    assert mask[0, 0] == 0
    empty, bad = rasterize_polygon(np.zeros((2, 2)), 32, 32)
    assert not bad
    assert empty.sum() == 0


def test_window_contains_annotations_and_rejects_gaps():
    window = plan_window(40, ed=5, es=15, clip_len=16, rng=np.random.default_rng(0))
    assert window is not None
    assert 0 <= window["ed_index"] < 16
    assert 0 <= window["es_index"] < 16
    assert window["start"] + window["ed_index"] == 5
    short = plan_window(10, ed=1, es=8, clip_len=16)
    assert short["pad"] == 6
    assert short["ed_index"] == 1
    assert plan_window(40, ed=0, es=20, clip_len=16) is None
    assert plan_window(6, ed=1, es=3, clip_len=16, min_frames=8) is None


def test_audit_rules():
    assert classify_case(file_exists=False, ef=50, n_frames=30, frames=[])["reason"] == "missing_file"
    assert classify_case(file_exists=True, ef=None, n_frames=30, frames=[])["reason"] == "bad_ef"
    assert classify_case(file_exists=True, ef=40, n_frames=5, frames=[(1, 100)])["reason"] == "too_short"
    oob = classify_case(file_exists=True, ef=40, n_frames=20, frames=[(50, 100)])
    assert oob["status"] == "critic_only" and oob["reason"] == "tracing_oob"
    one = classify_case(file_exists=True, ef=40, n_frames=20, frames=[(4, 80)])
    assert one["status"] == "critic_only" and one["reason"] == "one_tracing"
    far = classify_case(file_exists=True, ef=40, n_frames=40, frames=[(1, 200), (30, 80)])
    assert far["status"] == "critic_only" and far["reason"] == "ed_es_too_far"
    good = classify_case(file_exists=True, ef=35, n_frames=30, frames=[(3, 180), (12, 70)])
    assert good["status"] == "dit"
    assert good["ed"] == 3
    assert good["es"] == 12
    degenerate = classify_case(file_exists=True, ef=35, n_frames=30, frames=[(3, 1.0), (8, 2.0)])
    assert degenerate["reason"] == "degenerate"


def test_ratio_keeps_every_minority_row():
    from haemodiff.train.train_critic import majority_indices

    labels = np.array([1, 1] + [0] * 20)
    chosen = majority_indices(labels, "1:5", seed=0)
    assert int((labels[chosen] == 1).sum()) == 2
    assert int((labels[chosen] == 0).sum()) == 10
    assert len(majority_indices(labels, "natural", seed=0)) == len(labels)


def test_empty_collate_has_a_zero_batch():
    from haemodiff.data.dataset import collate_clips

    empty = collate_clips([])
    assert empty["video"].shape[0] == 0
    assert empty["filename"] == []


def test_synthetic_sources_cannot_be_test_ids():
    assert_no_test_leak(["train-a"], {"test-b"})
    try:
        assert_no_test_leak(["test-b"], {"test-b"})
    except RuntimeError as error:
        assert "test-b" in str(error)
    else:
        raise AssertionError("expected a test-id leak to fail")
