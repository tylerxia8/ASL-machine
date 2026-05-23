"""Load sign clips from disk (no PyTorch dependency)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NUM_FRAMES = 24
HEIGHT = 160
WIDTH = 160


def load_clip(path: Path) -> np.ndarray:
    if not path.is_absolute():
        root = Path(__file__).resolve().parent.parent
        path = root / path
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)["frames"]
        frames = np.array(raw, dtype=np.float32)
        if frames.ndim != 4:
            raise ValueError(f"Bad json clip shape {frames.shape}")
    else:
        data = np.load(path)
        frames = data["frames"].astype(np.float32)
    if frames.ndim == 4 and frames.shape[0] != NUM_FRAMES:
        idx = np.linspace(0, frames.shape[0] - 1, NUM_FRAMES).astype(int)
        frames = frames[idx]
    if frames.max() > 1.0:
        frames = frames / 255.0
    return frames.astype(np.float32)


def downsample_clip(frames: np.ndarray, t: int = 8, h: int = 32, w: int = 32) -> np.ndarray:
    """Reduce clip size for memory-efficient lite training."""
    ti = np.linspace(0, frames.shape[0] - 1, t).astype(int)
    f = frames[ti]
    sh = max(1, f.shape[1] // h)
    sw = max(1, f.shape[2] // w)
    small = f[:, ::sh, ::sw, :][:t, :h, :w, :]
    if small.shape[1] != h or small.shape[2] != w:
        out = np.zeros((t, h, w, 3), dtype=np.float32)
        for i in range(t):
            out[i] = np.array(
                [[f[i, min(y * sh, f.shape[1] - 1), min(x * sw, f.shape[2] - 1), :] for x in range(w)] for y in range(h)],
                dtype=np.float32,
            )
        return out
    return small.astype(np.float32)


def clip_feature_vector(path: Path, t: int = 8, h: int = 32, w: int = 32) -> np.ndarray:
    return downsample_clip(load_clip(path), t, h, w).flatten()
