"""Video file resolution and decoding."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def resolve_video_path(root: Path, filename: str) -> Path | None:
    stem = Path(filename).name
    names = [stem]
    if not stem.lower().endswith(".avi"):
        names.append(stem + ".avi")
    else:
        names.append(stem[:-4])
    folders = [root / "Videos", root / "videos", root]
    for folder in folders:
        for name in names:
            candidate = folder / name
            if candidate.is_file():
                return candidate
    return None


def read_grayscale_video(path: Path) -> np.ndarray:
    """Return ``(T, H, W)`` uint8 frames."""
    try:
        import torchvision

        video, _, _ = torchvision.io.read_video(str(path), output_format="TCHW")
        frames = video.detach().cpu().numpy()
        if frames.ndim != 4 or frames.shape[0] == 0:
            raise RuntimeError(f"No frames in {path}")
        if frames.shape[1] >= 3:
            gray = (
                0.2989 * frames[:, 2] + 0.5870 * frames[:, 1] + 0.1140 * frames[:, 0]
            )
        else:
            gray = frames[:, 0]
        return np.clip(gray, 0, 255).astype(np.uint8)
    except Exception:
        return _read_with_cv2(path)


def _read_with_cv2(path: Path) -> np.ndarray:
    import cv2

    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"Could not decode {path}")
    return np.stack(frames).astype(np.uint8)
