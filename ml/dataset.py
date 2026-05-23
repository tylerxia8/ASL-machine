from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from clip_io import load_clip

NUM_FRAMES = 24
HEIGHT = 160
WIDTH = 160


class SignClipDataset(Dataset):
    def __init__(self, manifest_path: Path, split: str, label_to_idx: dict[str, int]):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        self.items = [r for r in manifest["clips"] if r["split"] == split]
        self.label_to_idx = label_to_idx

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        row = self.items[i]
        frames = load_clip(Path(row["path"]))
        x = torch.from_numpy(frames).permute(3, 0, 1, 2)
        y = self.label_to_idx[row["sign_id"]]
        return x, y
