from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from clip_io import load_clip

NUM_FRAMES = 24
HEIGHT = 160
WIDTH = 160


# Augmentations applied at train-time only. Each value is bounded for ASL
# safety — overaggressive augs can change a sign's meaning.
#
# Deliberately OMITTED augmentations (and why):
# - Horizontal flip: some ASL signs depend on dominant-hand convention
#   (right-handed vs left-handed); a flipped HELLO can read as a different
#   gesture to learners. Flipping at training time would teach the model
#   to be invariant when the world isn't.
# - Color hue jitter: not relevant for our 25-sign roster (no color signs).
#   Could be re-added if the trained vocab expands to red/blue/green/etc.
# - Aggressive random crops (>10%): risks cutting the hands out of frame.
AUG_DEFAULTS = {
    "temporal_jitter": 2,        # ± frames when subsampling to NUM_FRAMES
    "brightness_jitter": 0.10,   # ± multiplicative factor on RGB
    "contrast_jitter": 0.10,     # ± multiplicative around the mean
    "spatial_crop_max": 0.04,    # max fraction trimmed off each side, then resized
    # NB: was 0.08 in v5/v6; halved because aggressive spatial crops shift the
    # hand centroid enough that the model can't pick up location-sensitive
    # signs (where vs what, please vs sorry). Combined with the v5/v6
    # mode-collapse evidence we err on the side of less aug.
}


def _temporal_subsample(frames: np.ndarray, n: int, jitter: int = 0,
                        rng: random.Random | None = None) -> np.ndarray:
    """Pick `n` evenly-spaced frames, optionally with random ±jitter offsets."""
    t = frames.shape[0]
    if t == n and jitter == 0:
        return frames
    if t <= n:
        # Repeat the last frame to reach n (rare path; clips should always be ≥ n).
        pad = np.repeat(frames[-1:], n - t, axis=0) if t < n else np.empty((0, *frames.shape[1:]))
        return np.concatenate([frames, pad], axis=0)
    idx = np.linspace(0, t - 1, n)
    if jitter and rng is not None:
        offsets = np.array([rng.randint(-jitter, jitter) for _ in range(n)])
        idx = np.clip(idx + offsets, 0, t - 1)
    return frames[idx.astype(int)]


def _brightness_contrast(frames: np.ndarray, b_jit: float, c_jit: float,
                         rng: random.Random) -> np.ndarray:
    """Per-clip (not per-frame) brightness + contrast jitter, applied uniformly."""
    if b_jit <= 0 and c_jit <= 0:
        return frames
    out = frames
    if b_jit > 0:
        factor = 1.0 + rng.uniform(-b_jit, b_jit)
        out = np.clip(out * factor, 0.0, 1.0)
    if c_jit > 0:
        factor = 1.0 + rng.uniform(-c_jit, c_jit)
        mean = out.mean(axis=(1, 2, 3), keepdims=True) if out.ndim == 4 else out.mean()
        out = np.clip((out - mean) * factor + mean, 0.0, 1.0)
    return out.astype(np.float32)


def _spatial_crop(frames: np.ndarray, max_frac: float, rng: random.Random) -> np.ndarray:
    """Random per-side crop then resize back to (HEIGHT, WIDTH). Same crop across all T frames."""
    if max_frac <= 0:
        return frames
    import cv2
    h, w = frames.shape[1], frames.shape[2]
    top = rng.randint(0, int(h * max_frac))
    bottom = rng.randint(0, int(h * max_frac))
    left = rng.randint(0, int(w * max_frac))
    right = rng.randint(0, int(w * max_frac))
    if top + bottom >= h or left + right >= w:
        return frames
    cropped = frames[:, top : h - bottom, left : w - right, :]
    if cropped.shape[1] == h and cropped.shape[2] == w:
        return frames
    out = np.empty((cropped.shape[0], HEIGHT, WIDTH, 3), dtype=np.float32)
    for i in range(cropped.shape[0]):
        out[i] = cv2.resize(cropped[i], (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    return out


class SignClipDataset(Dataset):
    def __init__(self, manifest_path: Path, split: str, label_to_idx: dict[str, int],
                 augment: bool | None = None, aug_config: dict | None = None,
                 seed: int = 0):
        """
        augment: if True, apply temporal jitter + brightness/contrast + spatial crop.
                 Defaults to True only for the 'train' split. Val/test are never augmented.
        """
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        self.items = [r for r in manifest["clips"] if r["split"] == split]
        self.label_to_idx = label_to_idx
        self.augment = (split == "train") if augment is None else augment
        self.aug = {**AUG_DEFAULTS, **(aug_config or {})}
        self._rng_seed = seed

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        row = self.items[i]
        frames = load_clip(Path(row["path"]))  # (T_raw, H, W, 3) float32 in [0,1]

        if self.augment:
            # Deterministic-ish per-(epoch, item) — torch DataLoader shuffles indices,
            # so a fresh Random keyed on idx + a per-instance seed is fine.
            rng = random.Random(self._rng_seed ^ (i * 2654435761 & 0xFFFFFFFF))
            frames = _temporal_subsample(frames, NUM_FRAMES, self.aug["temporal_jitter"], rng)
            frames = _brightness_contrast(frames, self.aug["brightness_jitter"],
                                          self.aug["contrast_jitter"], rng)
            frames = _spatial_crop(frames, self.aug["spatial_crop_max"], rng)
        else:
            frames = _temporal_subsample(frames, NUM_FRAMES, 0)

        # Ensure shape (NUM_FRAMES, H, W, 3), then permute to (3, T, H, W) for Conv3d.
        if frames.shape[1] != HEIGHT or frames.shape[2] != WIDTH:
            # Defensive: clip_io should have already normalized, but the spatial-crop
            # path can resize. Re-resize if needed.
            import cv2
            tmp = np.empty((frames.shape[0], HEIGHT, WIDTH, 3), dtype=np.float32)
            for k in range(frames.shape[0]):
                tmp[k] = cv2.resize(frames[k], (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
            frames = tmp
        x = torch.from_numpy(frames).permute(3, 0, 1, 2)
        y = self.label_to_idx[row["sign_id"]]
        return x, y
