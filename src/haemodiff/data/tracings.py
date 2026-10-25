"""LV tracing polygons and raster masks for EchoNet volume tracings."""

from __future__ import annotations

import numpy as np


def polygon_from_segments(segments: np.ndarray) -> np.ndarray:
    """Build a closed wall polygon from ``(N, 4)`` rows of x1, y1, x2, y2."""
    if segments.ndim != 2 or segments.shape[1] != 4 or len(segments) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    left = segments[:, 0:2]
    right = segments[:, 2:4][::-1]
    return np.concatenate([left, right], axis=0)


def polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def scale_points(points: np.ndarray, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> np.ndarray:
    if len(points) == 0:
        return points.copy()
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    scaled = points.astype(np.float64).copy()
    scaled[:, 0] *= dst_w / max(src_w, 1)
    scaled[:, 1] *= dst_h / max(src_h, 1)
    return scaled


def rasterize_polygon(points: np.ndarray, height: int, width: int, min_area: float = 10.0) -> tuple[np.ndarray, bool]:
    """Even-odd fill. Returns a float mask and whether the polygon is usable."""
    mask = np.zeros((height, width), dtype=np.float32)
    if len(points) < 3 or polygon_area(points) < min_area:
        return mask, False
    xs = points[:, 0]
    ys = points[:, 1]
    minx = max(int(np.floor(xs.min())), 0)
    maxx = min(int(np.ceil(xs.max())), width - 1)
    miny = max(int(np.floor(ys.min())), 0)
    maxy = min(int(np.ceil(ys.max())), height - 1)
    if maxx <= minx or maxy <= miny:
        return mask, False
    yy, xx = np.mgrid[miny : maxy + 1, minx : maxx + 1]
    inside = _points_in_polygon(xx.astype(np.float64), yy.astype(np.float64), points)
    mask[miny : maxy + 1, minx : maxx + 1] = inside.astype(np.float32)
    ok = float(mask.sum()) >= min_area
    if not ok:
        mask[:] = 0
    return mask, ok


def _points_in_polygon(x: np.ndarray, y: np.ndarray, poly: np.ndarray) -> np.ndarray:
    inside = np.zeros(x.shape, dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersects = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        )
        inside ^= intersects
        j = i
    return inside


def plan_window(
    n_frames: int,
    ed: int,
    es: int,
    clip_len: int = 16,
    min_frames: int = 8,
    rng: np.random.Generator | None = None,
) -> dict | None:
    """Return a 16-frame window that contains both annotated frames, or None."""
    if n_frames < min_frames:
        return None
    if ed < 0 or es < 0 or ed >= n_frames or es >= n_frames:
        return None
    if abs(int(ed) - int(es)) >= clip_len:
        return None
    if n_frames < clip_len:
        return {
            "start": 0,
            "source_len": int(n_frames),
            "pad": int(clip_len - n_frames),
            "ed_index": int(ed),
            "es_index": int(es),
        }
    lo = max(0, max(ed, es) - clip_len + 1)
    hi = min(min(ed, es), n_frames - clip_len)
    if lo > hi:
        return None
    if rng is None:
        start = int(lo)
    else:
        start = int(rng.integers(lo, hi + 1))
    return {
        "start": start,
        "source_len": clip_len,
        "pad": 0,
        "ed_index": int(ed - start),
        "es_index": int(es - start),
    }


def place_clip_masks(
    clip_len: int,
    height: int,
    width: int,
    ed_index: int,
    es_index: int,
    ed_mask: np.ndarray,
    es_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Masks are empty except on the annotated window indices. Valid flag is 0 elsewhere."""
    masks = np.zeros((clip_len, height, width), dtype=np.float32)
    valid = np.zeros((clip_len,), dtype=np.float32)
    if 0 <= ed_index < clip_len:
        masks[ed_index] = ed_mask
        valid[ed_index] = 1.0
    if 0 <= es_index < clip_len:
        masks[es_index] = es_mask
        valid[es_index] = 1.0
    return masks, valid


def anchor_window(
    n_frames: int,
    anchor: int,
    clip_len: int = 16,
    min_frames: int = 8,
    rng: np.random.Generator | None = None,
) -> dict | None:
    """Contiguous clip that contains ``anchor``. Used when ED and ES cannot share 16 frames."""
    if n_frames < min_frames or anchor < 0 or anchor >= n_frames:
        return None
    if n_frames <= clip_len:
        return {"start": 0, "source_len": int(n_frames), "pad": int(clip_len - n_frames)}
    lo = max(0, int(anchor) - clip_len + 1)
    hi = min(int(anchor), n_frames - clip_len)
    if lo > hi:
        return None
    start = int(lo if rng is None else rng.integers(lo, hi + 1))
    return {"start": start, "source_len": clip_len, "pad": 0}


def repeat_pad(frames: np.ndarray, clip_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Tile frames up to ``clip_len``. The pad mask is true on tiles after the first copy."""
    source_len = int(frames.shape[0])
    if source_len == clip_len:
        pad = np.zeros((clip_len,), dtype=bool)
        return frames, pad
    if source_len > clip_len:
        raise ValueError("repeat_pad expects a short clip")
    reps = int(np.ceil(clip_len / source_len))
    tiled = np.concatenate([frames] * reps, axis=0)[:clip_len]
    pad = np.zeros((clip_len,), dtype=bool)
    pad[source_len:] = True
    return tiled, pad
