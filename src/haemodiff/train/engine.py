"""Shared checkpointing, loaders, and epoch loops."""

from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from haemodiff.config import (
    get_device,
    resolve_run_dir,
    save_config_copy,
    select_batch_size,
    set_seed,
    stage_complete,
    write_complete,
)
from haemodiff.data.audit import (
    assert_full_training_split,
    build_index,
    count_status,
    load_tracing_groups,
    write_audit,
)
from haemodiff.data.dataset import EchoClipDataset, collate_clips


def prepare(cfg: dict) -> tuple[Path, torch.device, int]:
    set_seed(int(cfg["seed"]))
    device = get_device(bool(cfg.get("require_gpu", False)))
    run_dir = resolve_run_dir(cfg)
    save_config_copy(run_dir, cfg)
    batch_size = select_batch_size(cfg.get("optim", {}).get("batch_size"), bool(cfg.get("require_gpu", False)))
    return run_dir, device, batch_size


def load_records(cfg: dict, run_dir: Path):
    from haemodiff.config import find_echonet_root

    root = find_echonet_root(cfg)
    data = cfg["data"]
    records = build_index(root, int(data["clip_len"]), int(data["min_frames"]), int(data["size"]))
    write_audit(records, run_dir / "tracing_audit.csv")
    groups = load_tracing_groups(root / "VolumeTracings.csv")
    print("Tracing audit counts:")
    for key, value in sorted(count_status(records).items()):
        print(f"  {key}: {value}")
    return records, groups


def _geometry(dataset) -> tuple[int, int]:
    seen = dataset
    while isinstance(seen, Subset):
        seen = seen.dataset
    if isinstance(seen, ConcatDataset):
        seen = seen.datasets[0]
        while isinstance(seen, Subset):
            seen = seen.dataset
    return int(getattr(seen, "clip_len", 16)), int(getattr(seen, "size", 112))


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    workers = 0 if os.name == "nt" else num_workers
    clip_len, size = _geometry(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=partial(collate_clips, clip_len=clip_len, size=size),
        pin_memory=torch.cuda.is_available(),
    )


def make_split_dataset(records, groups, cfg: dict, split: str, role: str) -> EchoClipDataset:
    data = cfg["data"]
    return EchoClipDataset(
        records,
        groups,
        split=split,
        role=role,
        clip_len=int(data["clip_len"]),
        min_frames=int(data["min_frames"]),
        size=int(data["size"]),
        clinical_seed=int(cfg["seed"]) + (0 if split == "TRAIN" else 101),
        window_seed=int(cfg["seed"]) + 17,
        severe_ef=float(data["severe_ef"]),
        reduced_ef=float(data["reduced_ef"]),
    )


def require_full_dit_train(records, dataset: EchoClipDataset) -> None:
    assert_full_training_split(records, dataset.records)


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def append_history(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def run_stage(
    *,
    run_dir: Path,
    stage: str,
    epochs: int,
    patience: int,
    start_epoch: int,
    state: dict,
    train_one_epoch: Callable[[int], float],
    validate: Callable[[], float],
    save_state: Callable[[int, float, bool], None],
) -> None:
    if stage_complete(run_dir, stage):
        print(f"Skipping completed stage {stage}")
        return
    best = float(state.get("best", float("inf")))
    bad = int(state.get("bad", 0))
    epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(epoch)
        val_loss = validate()
        improved = val_loss < best - 1e-6
        if improved:
            best = val_loss
            bad = 0
        else:
            bad += 1
        save_state(epoch, best, improved)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "best": best, "bad": bad}
        append_history(run_dir / f"{stage}.history.jsonl", row)
        print(f"{stage} epoch {epoch}: train {train_loss:.4f} val {val_loss:.4f}")
        if bad >= patience:
            print(f"{stage} early stop at epoch {epoch}")
            break
    write_complete(run_dir, stage, {"stage": stage, "best": best, "epochs_ran": epoch + 1})
