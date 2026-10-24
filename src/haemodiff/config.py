"""Configuration and run-directory helpers."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} must be a mapping.")
    return cfg


def set_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_batch_size(configured: int | None, require_gpu: bool) -> int:
    import torch

    if configured is not None:
        return int(configured)
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError("GPU required. On Kaggle, set the notebook accelerator to GPU.")
    if not torch.cuda.is_available():
        return 1
    free_bytes, _total = torch.cuda.mem_get_info()
    if free_bytes >= 24 * (1024**3):
        return 2
    return 1


def get_device(require_gpu: bool):
    import torch

    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError("GPU required. On Kaggle, set the notebook accelerator to GPU.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _kaggle_working() -> Path | None:
    path = Path("/kaggle/working")
    return path if path.exists() else None


def resolve_run_dir(cfg: dict[str, Any]) -> Path:
    """Checkpoints live under /kaggle/working when that mount exists."""
    run_id = str(cfg.get("run_id", "haemodiff"))
    working = _kaggle_working()
    if working is not None and cfg.get("output_root") is None:
        primary = working / "haemodiff-runs" / run_id
    else:
        root = Path(cfg.get("output_root") or "runs")
        primary = root / run_id
    primary.mkdir(parents=True, exist_ok=True)
    _restore_previous_output(primary, run_id)
    return primary


def _restore_previous_output(primary: Path, run_id: str) -> None:
    """Copy a reattached Kaggle output into working storage when this session is empty."""
    if any(primary.glob("*.pt")) or any(primary.glob("*.complete.json")):
        return
    search_root = Path("/kaggle/input")
    if not search_root.exists():
        return
    for candidate in search_root.rglob(run_id):
        if not candidate.is_dir():
            continue
        if not (any(candidate.glob("*.pt")) or any(candidate.glob("*.complete.json"))):
            continue
        for item in candidate.iterdir():
            dest = primary / item.name
            if dest.exists():
                continue
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
        break


def stage_complete(run_dir: Path, stage: str) -> bool:
    return (run_dir / f"{stage}.complete.json").is_file()


def write_complete(run_dir: Path, stage: str, payload: dict[str, Any]) -> None:
    path = run_dir / f"{stage}.complete.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_config_copy(run_dir: Path, cfg: dict[str, Any]) -> None:
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def find_echonet_root(cfg: dict[str, Any]) -> Path:
    configured = cfg.get("data", {}).get("echonet_root")
    env = os.environ.get("ECHONET_ROOT")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))
    if env:
        candidates.append(Path(env))
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(kaggle_input.rglob("FileList.csv"))
    for candidate in candidates:
        root = candidate.parent if candidate.name == "FileList.csv" else candidate
        if (root / "FileList.csv").is_file() and (root / "VolumeTracings.csv").is_file():
            return root
    raise FileNotFoundError(
        "EchoNet-Dynamic was not found. Attach it on Kaggle so FileList.csv "
        "and VolumeTracings.csv are under /kaggle/input, or set ECHONET_ROOT."
    )
