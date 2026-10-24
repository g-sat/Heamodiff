"""EchoNet index rules. No video pixels are required."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from haemodiff.data.tracings import polygon_area, polygon_from_segments


def _as_float(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def load_filelist(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for raw in reader:
            filename = (raw.get("FileName") or "").strip()
            ef = _as_float((raw.get("EF") or "").strip())
            n_frames = _as_float((raw.get("NumberOfFrames") or "").strip())
            height = _as_float((raw.get("FrameHeight") or "112").strip()) or 112.0
            width = _as_float((raw.get("FrameWidth") or "112").strip()) or 112.0
            split = (raw.get("Split") or "").strip().upper()
            rows.append(
                {
                    "filename": filename,
                    "ef": ef,
                    "n_frames": None if n_frames is None else int(n_frames),
                    "frame_height": int(height),
                    "frame_width": int(width),
                    "split": split,
                }
            )
    return rows


def load_tracing_groups(path: Path) -> dict[str, dict[int, np.ndarray]]:
    grouped: dict[str, dict[int, list[list[float]]]] = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            filename = (raw.get("FileName") or "").strip()
            frame = _as_float((raw.get("Frame") or "").strip())
            coords = [
                _as_float((raw.get(key) or "").strip())
                for key in ("X1", "Y1", "X2", "Y2")
            ]
            if not filename or frame is None or any(value is None for value in coords):
                continue
            grouped.setdefault(filename, {}).setdefault(int(frame), []).append(
                [float(value) for value in coords]
            )
    return {
        filename: {frame: np.asarray(segs, dtype=np.float64) for frame, segs in frames.items()}
        for filename, frames in grouped.items()
    }


def classify_case(
    *,
    file_exists: bool,
    ef: float | None,
    n_frames: int | None,
    frames: list[tuple[int, float]],
    clip_len: int = 16,
    min_frames: int = 8,
) -> dict:
    """Apply the tracing rules.

    ``frames`` is a list of ``(frame_index, area)`` for candidate tracings.
    Status is ``dit``, ``critic_only``, or ``drop``.
    """
    if not file_exists:
        return _decision("drop", "missing_file", None, None)
    if ef is None:
        return _decision("drop", "bad_ef", None, None)
    if n_frames is None or n_frames < min_frames:
        return _decision("drop", "too_short", None, None)

    valid: list[tuple[int, float]] = []
    saw_oob = False
    saw_degenerate = False
    for frame, area in frames:
        if frame < 0 or frame >= n_frames:
            saw_oob = True
            continue
        if area < 10.0:
            saw_degenerate = True
            continue
        valid.append((int(frame), float(area)))

    if len(valid) >= 2:
        ordered = sorted(valid, key=lambda item: item[1])
        es, ed = ordered[0][0], ordered[-1][0]
        if abs(ed - es) >= clip_len:
            return _decision("critic_only", "ed_es_too_far", ed, es)
        return _decision("dit", "ok", ed, es)
    if len(valid) == 1:
        return _decision("critic_only", "one_tracing", valid[0][0], None)
    if saw_oob:
        return _decision("critic_only", "tracing_oob", None, None)
    if saw_degenerate:
        return _decision("critic_only", "degenerate", None, None)
    return _decision("critic_only", "no_tracing", None, None)


def _decision(status: str, reason: str, ed: int | None, es: int | None) -> dict:
    return {"status": status, "reason": reason, "ed": ed, "es": es}


def tracing_areas(
    segments_by_frame: dict[int, np.ndarray],
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
) -> list[tuple[int, float]]:
    from haemodiff.data.tracings import scale_points

    found = []
    for frame, segments in segments_by_frame.items():
        points = scale_points(polygon_from_segments(segments), src_hw, dst_hw)
        found.append((int(frame), polygon_area(points)))
    return found


def build_index(root: Path, clip_len: int = 16, min_frames: int = 8, size: int = 112) -> list[dict]:
    from haemodiff.data.video import resolve_video_path

    root = Path(root)
    studies = load_filelist(root / "FileList.csv")
    tracings = load_tracing_groups(root / "VolumeTracings.csv")
    records = []
    for row in studies:
        path = resolve_video_path(root, row["filename"])
        groups = tracings.get(row["filename"], {})
        src_hw = (row["frame_height"], row["frame_width"])
        areas = tracing_areas(groups, src_hw, (size, size))
        decision = classify_case(
            file_exists=path is not None,
            ef=row["ef"],
            n_frames=row["n_frames"],
            frames=areas,
            clip_len=clip_len,
            min_frames=min_frames,
        )
        records.append(
            {
                **row,
                "video_path": "" if path is None else str(path),
                "status": decision["status"],
                "reason": decision["reason"],
                "ed": decision["ed"],
                "es": decision["es"],
            }
        )
    return records


def write_audit(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "filename",
        "split",
        "status",
        "reason",
        "ef",
        "n_frames",
        "ed",
        "es",
        "video_path",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def assert_full_training_split(records: list[dict], used: list[dict]) -> None:
    expected = [row for row in records if row["split"] == "TRAIN" and row["status"] == "dit"]
    if len(expected) == 0:
        raise RuntimeError("Indexed training split is empty after the tracing rules.")
    if len(used) < len(expected):
        raise RuntimeError(
            f"Training split has {len(used)} clips, smaller than the full post-audit "
            f"training split of {len(expected)}."
        )


def count_status(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in records:
        key = f"{row['split']}:{row['status']}:{row['reason']}"
        counts[key] = counts.get(key, 0) + 1
    return counts
