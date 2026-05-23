"""Generate synthetic clip data for pipeline testing (replace with real recordings for pilot)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from dataset import HEIGHT, NUM_FRAMES, WIDTH  # noqa: E402

CLIPS_DIR = ROOT / "ml" / "data" / "clips"
VOCAB_PATH = ROOT / "content" / "vocabulary.csv"


def read_sign_ids(limit: int | None) -> list[str]:
    import csv

    signs = []
    with open(VOCAB_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            signs.append(row["sign_id"])
            if limit and len(signs) >= limit:
                break
    return signs


def make_clip(sign_index: int, clip_index: int, signer_index: int) -> np.ndarray:
    """Distinct patterns per sign for trainable synthetic separation."""
    rng = np.random.default_rng(sign_index * 1000 + clip_index * 17 + signer_index * 3)
    t = np.linspace(0, 1, NUM_FRAMES)
    base = sign_index / 100.0
    frames = np.zeros((NUM_FRAMES, HEIGHT, WIDTH, 3), dtype=np.float32)
    cx, cy = WIDTH // 2, HEIGHT // 2
    for i, _ in enumerate(t):
        phase = base + 0.1 * np.sin(2 * np.pi * (t[i] + clip_index * 0.05))
        radius = int(20 + 15 * sign_index % 7 + 5 * np.sin(phase * 10))
        y, x = np.ogrid[:HEIGHT, :WIDTH]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
        color = np.array(
            [
                0.3 + 0.5 * (sign_index % 5) / 5,
                0.2 + 0.4 * (signer_index % 3) / 3,
                0.4 + 0.3 * np.sin(phase),
            ],
            dtype=np.float32,
        )
        noise = rng.normal(0, 0.02, (HEIGHT, WIDTH, 3)).astype(np.float32)
        frame = np.full((HEIGHT, WIDTH, 3), 0.1, dtype=np.float32) + noise
        frame[mask] = color
        frames[i] = np.clip(frame, 0, 1)
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signs", type=int, default=20)
    parser.add_argument("--clips-per-sign", type=int, default=40)
    parser.add_argument("--signers", type=int, default=2)
    args = parser.parse_args()

    sign_ids = read_sign_ids(args.signs)
    clips_meta = []
    for si, sign_id in enumerate(sign_ids):
        for signer_i in range(args.signers):
            signer_id = f"signer_{chr(ord('a') + signer_i)}"
            for ci in range(args.clips_per_sign):
                out_dir = CLIPS_DIR / sign_id / signer_id
                out_dir.mkdir(parents=True, exist_ok=True)
                rel = f"ml/data/clips/{sign_id}/{signer_id}/clip_{ci:04d}.npz"
                path = ROOT / rel
                frames = make_clip(si, ci, signer_i)
                np.savez_compressed(path, frames=frames)
                split = "test" if signer_i == args.signers - 1 and ci >= args.clips_per_sign * 0.8 else (
                    "val" if ci >= args.clips_per_sign * 0.7 else "train"
                )
                if signer_i == args.signers - 1:
                    split = "test" if ci < args.clips_per_sign // 5 else "train" if ci < args.clips_per_sign * 0.7 else "val"
                clips_meta.append(
                    {
                        "path": str(path).replace("\\", "/"),
                        "sign_id": sign_id,
                        "signer_id": signer_id,
                        "split": split,
                    }
                )

    manifest = {"clips": clips_meta, "sign_ids": sign_ids}
    manifest_path = ROOT / "ml" / "data" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(clips_meta)} clips for {len(sign_ids)} signs -> {manifest_path}")


if __name__ == "__main__":
    main()
