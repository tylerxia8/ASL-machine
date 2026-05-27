"""Smoke test for make_contact_sheets.py."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from make_contact_sheets import render_sheet


def _clip(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frames = np.zeros((24, 160, 160, 3), dtype=np.float32)
    for i in range(frames.shape[0]):
        frames[i, :, :, 0] = i / frames.shape[0]
        frames[i, :, :, 1] = rng.uniform(0.1, 0.8)
        frames[i, 40:120, 50:110, 2] = 0.9
    return frames


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        clips_dir = root / "clips"
        out_dir = root / "out"
        rows = []
        for i, sign_id in enumerate(("hello", "where")):
            path = clips_dir / sign_id / "signer_test" / "clip_0000.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, frames=_clip(i))
            rows.append(
                {
                    "path": str(path),
                    "sign_id": sign_id,
                    "signer_id": "signer_test",
                    "split": "test",
                }
            )
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"clips": rows, "sign_ids": ["hello", "where"]}), encoding="utf-8")

        out_path = out_dir / "sheet.jpg"
        count, raw_matches = render_sheet(rows, out_path, root / "incoming", frames_per_clip=4)
        assert count == 2
        assert raw_matches == 0
        assert out_path.exists()
        with Image.open(out_path) as img:
            assert img.width > 0
            assert img.height > 0
            assert img.mode == "RGB"
    print("contact sheet smoke ok")


if __name__ == "__main__":
    main()
