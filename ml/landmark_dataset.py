from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

NUM_FRAMES = 24
HAND_FEATURES = 132


def feature_path_for_clip(clip_path: str, feature_dir: Path) -> Path:
    safe = clip_path.replace("\\", "/").replace("/", "__")
    if not safe.endswith(".npz"):
        safe = f"{safe}.npz"
    return feature_dir / safe


class HandLandmarkDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        split: str,
        label_to_idx: dict[str, int],
        feature_dir: Path,
        augment: bool | None = None,
        noise_std: float = 0.01,
        dropout_prob: float = 0.05,
        seed: int = 0,
    ):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        rows = [r for r in manifest["clips"] if r["split"] == split]
        self.label_to_idx = label_to_idx
        self.feature_dir = feature_dir
        self.items = [r for r in rows if feature_path_for_clip(r["path"], feature_dir).exists()]
        self.missing_features_count = len(rows) - len(self.items)
        self.augment = (split == "train") if augment is None else augment
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.seed = seed

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        row = self.items[i]
        path = feature_path_for_clip(row["path"], self.feature_dir)
        data = np.load(path)
        features = data["features"].astype(np.float32)
        if features.shape != (NUM_FRAMES, HAND_FEATURES):
            raise ValueError(f"Bad landmark feature shape {features.shape} in {path}")

        if self.augment:
            rng = np.random.default_rng(self.seed ^ (i * 2654435761 & 0xFFFFFFFF))
            present = np.abs(features).sum(axis=1) > 0
            if self.noise_std > 0:
                noise = rng.normal(0.0, self.noise_std, size=features.shape).astype(np.float32)
                features = np.where(present[:, None], features + noise, features)
            if self.dropout_prob > 0:
                mask = rng.random(features.shape[0]) < self.dropout_prob
                features = features.copy()
                features[mask] = 0.0

        # Conv1d expects (features, time).
        x = torch.from_numpy(features).transpose(0, 1)
        y = self.label_to_idx[row["sign_id"]]
        return x, y
