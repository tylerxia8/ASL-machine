"""Import Capture-page downloads (webm or legacy json) into ml/data/clips as .npz."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INCOMING = ROOT / "ml" / "data" / "incoming"
CLIPS = ROOT / "ml" / "data" / "clips"
NUM_FRAMES, HEIGHT, WIDTH = 24, 160, 160
RESIZE_MODES = ("center_crop", "letterbox")


def _resample(frames: np.ndarray, target: int) -> np.ndarray:
    if frames.shape[0] == target:
        return frames
    idx = np.linspace(0, frames.shape[0] - 1, target).astype(int)
    return frames[idx]


def _resize_frame(frame: np.ndarray, mode: str) -> np.ndarray:
    import cv2

    h, w = frame.shape[:2]
    if mode == "center_crop":
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        frame = frame[y0 : y0 + side, x0 : x0 + side]
        return cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    if mode == "letterbox":
        scale = min(WIDTH / w, HEIGHT / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        out = np.zeros((HEIGHT, WIDTH, 3), dtype=resized.dtype)
        x0 = (WIDTH - new_w) // 2
        y0 = (HEIGHT - new_h) // 2
        out[y0 : y0 + new_h, x0 : x0 + new_w] = resized
        return out
    raise ValueError(f"Unknown resize mode {mode!r}; expected one of {RESIZE_MODES}")


def _normalize_frames(frames: np.ndarray, resize_mode: str = "center_crop") -> np.ndarray:
    """Resize to (HEIGHT, WIDTH), normalize to [0,1] float32."""
    out = np.empty((frames.shape[0], HEIGHT, WIDTH, 3), dtype=np.float32)
    for i, frame in enumerate(frames):
        resized = _resize_frame(frame, resize_mode)
        if resized.dtype != np.float32:
            resized = resized.astype(np.float32)
        if resized.max() > 1.0:
            resized = resized / 255.0
        out[i] = resized
    return out


def load_webm(path: Path, resize_mode: str = "center_crop") -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open {path.name}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No decodable frames in {path.name}")
    arr = np.stack(frames)
    arr = _resample(arr, NUM_FRAMES)
    return _normalize_frames(arr, resize_mode)


def load_json(path: Path) -> np.ndarray:
    raw = json.loads(path.read_text(encoding="utf-8"))["frames"]
    arr = np.array(raw, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"Bad json clip shape {arr.shape} in {path.name}")
    arr = _resample(arr, NUM_FRAMES)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr.astype(np.float32)


def infer_meta(path: Path, data: dict | None) -> tuple[str, str]:
    if data:
        sign_id = data.get("sign_id")
        signer_id = data.get("signer_id")
        if sign_id and signer_id:
            return sign_id, signer_id
    stem = path.stem
    m = re.match(r"^(.+)_(signer_[a-z0-9]+)_\d+$", stem)
    if m:
        return m.group(1), m.group(2)
    parts = stem.split("_")
    if len(parts) >= 3 and parts[-2].startswith("signer"):
        return "_".join(parts[:-2]), "_".join(parts[-2:])
    raise ValueError(f"Cannot infer sign_id/signer_id from {path.name}")


VIDEO_EXTS = (".webm", ".mp4", ".mov", ".mkv", ".avi")


def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resize-mode", choices=RESIZE_MODES, default="center_crop",
                        help="How raw videos are resized to square model frames.")
    args = parser.parse_args()

    INCOMING.mkdir(parents=True, exist_ok=True)
    CLIPS.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for ext in VIDEO_EXTS:
        files.extend(INCOMING.glob(f"*{ext}"))
    files.extend(INCOMING.glob("*.json"))  # legacy capture-page format
    files = sorted(files)
    if not files:
        print(f"No video files in {INCOMING}. Expected one of {VIDEO_EXTS} or *.json.",
              file=sys.stderr)
        raise SystemExit(2)

    print(f"Found {len(files)} input file(s) "
          f"({sum(1 for f in files if f.suffix in VIDEO_EXTS)} video, "
          f"{sum(1 for f in files if f.suffix == '.json')} legacy json).")

    imported = 0
    failed: list[tuple[str, str]] = []
    for path in files:
        try:
            if path.suffix == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                frames = load_json(path)
                sign_id, signer_id = infer_meta(path, data)
            else:
                # cv2.VideoCapture handles webm/mp4/mov/mkv/avi via FFmpeg.
                frames = load_webm(path, args.resize_mode)
                sign_id, signer_id = infer_meta(path, None)
            out_dir = CLIPS / sign_id / signer_id
            out_dir.mkdir(parents=True, exist_ok=True)
            existing = len(list(out_dir.glob("*.npz")))
            out_path = out_dir / f"clip_{existing:04d}.npz"
            np.savez_compressed(
                out_path,
                frames=frames,
                source_path=str(path.relative_to(ROOT)).replace("\\", "/"),
                resize_mode=args.resize_mode,
            )
            imported += 1
            print(f"Imported {path.name} -> {out_path.relative_to(ROOT)}")
        except Exception as e:
            failed.append((path.name, str(e)))
            print(f"FAILED {path.name}: {e}", file=sys.stderr)

    print(f"\nDone: {imported} imported, {len(failed)} failed.")
    if imported == 0:
        print("FATAL: 0 imports succeeded. Failures above show the underlying cause "
              "(missing codec, corrupt file, unparseable filename, etc.).", file=sys.stderr)
        raise SystemExit(3)
    if imported:
        print("Next: python ml/scripts/build_manifest.py --wave1 --signer-disjoint")


if __name__ == "__main__":
    main()
