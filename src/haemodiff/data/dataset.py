"""Clip datasets for diffusion and the evidential critic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from haemodiff.data.clinical import normalize_clinical, sample_clinical
from haemodiff.data.audit import tracing_lookup_key
from haemodiff.data.tracings import (
    anchor_window,
    place_clip_masks,
    plan_window,
    polygon_from_segments,
    rasterize_polygon,
    repeat_pad,
    scale_points,
)
from haemodiff.data.video import read_grayscale_video


def resize_frames(frames: np.ndarray, size: int) -> np.ndarray:
    tensor = torch.from_numpy(frames.astype(np.float32) / 255.0)[:, None]
    resized = torch.nn.functional.interpolate(
        tensor, size=(size, size), mode="bilinear", align_corners=False
    )
    return resized[:, 0].numpy()


class EchoClipDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        tracing_groups: dict,
        *,
        split: str,
        role: str,
        clip_len: int,
        min_frames: int,
        size: int,
        clinical_seed: int,
        window_seed: int,
        severe_ef: float,
        reduced_ef: float,
        cover_annotation: bool = False,
    ) -> None:
        if role not in {"dit", "critic"}:
            raise ValueError(role)
        chosen = []
        for row in records:
            if row["split"] != split:
                continue
            if role == "dit" and row["status"] != "dit":
                continue
            if role == "critic" and row["status"] == "drop":
                continue
            if row["ef"] is None:
                continue
            chosen.append(row)
        chosen.sort(key=lambda row: row["filename"])
        self.records = chosen
        self.tracing_groups = tracing_groups
        self.clip_len = clip_len
        self.min_frames = min_frames
        self.size = size
        self.window_seed = window_seed
        self.severe_ef = severe_ef
        self.reduced_ef = reduced_ef
        self.cover_annotation = cover_annotation
        raw = sample_clinical(len(chosen), clinical_seed)
        order = np.random.default_rng(clinical_seed).permutation(len(chosen))
        self.clinical = normalize_clinical(raw[order])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int | float]:
        row = self.records[index]
        frames = read_grayscale_video(Path(row["video_path"]))
        frames = resize_frames(frames, self.size)
        n_frames = int(frames.shape[0])
        ed = row["ed"]
        es = row["es"]
        window = None
        if ed is not None and es is not None:
            rng = np.random.default_rng(self.window_seed + index)
            window = plan_window(n_frames, int(ed), int(es), self.clip_len, self.min_frames, rng)
        if row["status"] == "dit" and window is None:
            raise RuntimeError(
                f"{row['filename']} is indexed for diffusion but ED/ES do not fit the decoded clip."
            )
        if window is None and self.cover_annotation:
            window = self._anchored_window(n_frames, ed, es, index)
        if window is None:
            if n_frames < self.min_frames:
                raise RuntimeError(f"{row['filename']} became too short at decode time.")
            source = frames[: self.clip_len]
            clip, pad = repeat_pad(source, self.clip_len)
            ed_index, es_index = -1, -1
            ed_mask = np.zeros((self.size, self.size), dtype=np.float32)
            es_mask = np.zeros_like(ed_mask)
        else:
            source = frames[window["start"] : window["start"] + window["source_len"]]
            clip, pad = repeat_pad(source, self.clip_len)
            ed_index = int(window["ed_index"])
            es_index = int(window["es_index"])
            if ed is None or es is None:
                ed_mask = np.zeros((self.size, self.size), dtype=np.float32)
                es_mask = np.zeros_like(ed_mask)
            else:
                ed_mask, es_mask = self._masks(row, int(ed), int(es))
        masks, valid = place_clip_masks(
            self.clip_len, self.size, self.size, ed_index, es_index, ed_mask, es_mask
        )
        ef = float(row["ef"])
        return {
            "video": torch.from_numpy(clip[:, None]),
            "mask": torch.from_numpy(masks[:, None]),
            "valid_mask": torch.from_numpy(valid),
            "pad": torch.from_numpy(pad),
            "ef": torch.tensor(ef, dtype=torch.float32),
            "severe": torch.tensor(float(ef < self.severe_ef), dtype=torch.float32),
            "reduced": torch.tensor(float(ef < self.reduced_ef), dtype=torch.float32),
            "clinical": torch.from_numpy(self.clinical[index]),
            "ed_index": ed_index,
            "es_index": es_index,
            "filename": row["filename"],
            "split": row["split"],
        }

    def _anchored_window(self, n_frames: int, ed, es, index: int) -> dict | None:
        if n_frames < self.min_frames:
            return None
        rng = np.random.default_rng(self.window_seed + index + 1)
        anchors = [int(frame) for frame in (ed, es) if frame is not None and 0 <= int(frame) < n_frames]
        anchor = int(anchors[int(rng.integers(0, len(anchors)))]) if anchors else int(rng.integers(0, n_frames))
        placed = anchor_window(n_frames, anchor, self.clip_len, self.min_frames, rng)
        if placed is None:
            return None
        start = int(placed["start"])
        stop = start + int(placed["source_len"])
        placed["ed_index"] = int(ed) - start if ed is not None and start <= int(ed) < stop else -1
        placed["es_index"] = int(es) - start if es is not None and start <= int(es) < stop else -1
        return placed

    def _masks(self, row: dict, ed: int, es: int) -> tuple[np.ndarray, np.ndarray]:
        groups = self.tracing_groups.get(tracing_lookup_key(row["filename"]), {})
        src = (int(row["frame_height"]), int(row["frame_width"]))
        dst = (self.size, self.size)
        return _one_mask(groups.get(ed), src, dst), _one_mask(groups.get(es), src, dst)


def _one_mask(segments: np.ndarray | None, src: tuple[int, int], dst: tuple[int, int]) -> np.ndarray:
    blank = np.zeros(dst, dtype=np.float32)
    if segments is None:
        return blank
    points = scale_points(polygon_from_segments(segments), src, dst)
    mask, ok = rasterize_polygon(points, dst[0], dst[1])
    return mask if ok else blank


def collate_clips(batch: list[dict], clip_len: int = 16, size: int = 112, clinical_dim: int = 5) -> dict:
    if not batch:
        return {
            "video": torch.zeros(0, clip_len, 1, size, size),
            "mask": torch.zeros(0, clip_len, 1, size, size),
            "valid_mask": torch.zeros(0, clip_len),
            "pad": torch.zeros(0, clip_len, dtype=torch.bool),
            "ef": torch.zeros(0),
            "severe": torch.zeros(0),
            "reduced": torch.zeros(0),
            "clinical": torch.zeros(0, clinical_dim),
            "ed_index": [],
            "es_index": [],
            "filename": [],
            "split": [],
        }
    out: dict = {}
    for key in ("video", "mask", "valid_mask", "pad", "ef", "severe", "reduced", "clinical"):
        out[key] = torch.stack([item[key] for item in batch])
    out["ed_index"] = [item["ed_index"] for item in batch]
    out["es_index"] = [item["es_index"] for item in batch]
    out["filename"] = [item["filename"] for item in batch]
    out["split"] = [item["split"] for item in batch]
    return out


def assert_no_test_leak(source_ids: list[str], test_ids: set[str]) -> None:
    leaked = sorted(set(source_ids) & set(test_ids))
    if leaked:
        raise RuntimeError(f"Synthetic samples point at test ids: {leaked[:5]}")
