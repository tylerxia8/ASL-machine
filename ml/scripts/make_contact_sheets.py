"""Render visual QA contact sheets for imported sign clips.

The main use is diagnosing whether Sem-Lex videos still contain the hands and
signing space after preprocessing. If matching raw videos are available in
`ml/data/incoming`, each sampled frame is shown as original-with-crop-box over
the imported frame. If raw videos are absent, the sheet still shows the model
input frames from `.npz` clips.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from clip_io import load_clip  # noqa: E402

VIDEO_EXTS = (".webm", ".mp4", ".mov", ".mkv", ".avi")
CELL_W = 180
CELL_H_PROCESSED = 160
CELL_H_RAW = 160
LABEL_H = 34
PAD = 10
BG = (245, 245, 242)
INK = (24, 24, 24)
MUTED = (96, 96, 96)
ACCENT = (220, 60, 45)


def _font(size: int = 13):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _to_image(frame: np.ndarray) -> Image.Image:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _letterbox(img: Image.Image, size: tuple[int, int], fill: tuple[int, int, int] = (20, 20, 20)) -> Image.Image:
    out = Image.new("RGB", size, fill)
    scale = min(size[0] / img.width, size[1] / img.height)
    new_size = (max(1, int(round(img.width * scale))), max(1, int(round(img.height * scale))))
    resized = img.resize(new_size, Image.Resampling.LANCZOS)
    out.paste(resized, ((size[0] - new_size[0]) // 2, (size[1] - new_size[1]) // 2))
    return out


def _crop_rect_in_letterbox(img: Image.Image, size: tuple[int, int]) -> tuple[int, int, int, int]:
    scale = min(size[0] / img.width, size[1] / img.height)
    new_w = max(1, int(round(img.width * scale)))
    new_h = max(1, int(round(img.height * scale)))
    ox = (size[0] - new_w) // 2
    oy = (size[1] - new_h) // 2
    side = min(img.width, img.height)
    x0 = (img.width - side) // 2
    y0 = (img.height - side) // 2
    return (
        ox + int(round(x0 * scale)),
        oy + int(round(y0 * scale)),
        ox + int(round((x0 + side) * scale)),
        oy + int(round((y0 + side) * scale)),
    )


def _resample_indices(length: int, target: int) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=int)
    return np.linspace(0, length - 1, target).astype(int)


def _read_video_frames(path: Path, target: int) -> list[Image.Image]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No decodable frames in {path}")
    idx = _resample_indices(len(frames), target)
    return [_to_image(frames[i]) for i in idx]


def _raw_video_index(video_dir: Path) -> dict[tuple[str, str, str], Path]:
    index: dict[tuple[str, str, str], Path] = {}
    if not video_dir.exists():
        return index
    pattern = re.compile(r"^(.+)_(signer_[A-Za-z0-9]+)_(\d+)$")
    for ext in VIDEO_EXTS:
        for path in video_dir.glob(f"*{ext}"):
            m = pattern.match(path.stem)
            if m:
                sign_id, signer_id, clip_num = m.groups()
                index[(sign_id, signer_id, f"{int(clip_num):04d}")] = path
    return index


def _clip_num_from_npz(path: Path) -> str | None:
    m = re.match(r"^clip_(\d+)$", path.stem)
    if not m:
        return None
    return f"{int(m.group(1)):04d}"


def _source_video_from_npz(path: Path) -> Path | None:
    try:
        with np.load(path) as data:
            if "source_path" not in data:
                return None
            raw_value = data["source_path"]
    except Exception:
        return None

    source = str(raw_value.item() if hasattr(raw_value, "item") else raw_value)
    if not source:
        return None
    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    return source_path if source_path.exists() else None


def _select_rows(manifest: dict, split: str, signs: set[str] | None, max_signs: int,
                 clips_per_sign: int) -> list[dict]:
    by_sign: dict[str, list[dict]] = defaultdict(list)
    for row in manifest.get("clips", []):
        if split != "all" and row.get("split") != split:
            continue
        if signs and row.get("sign_id") not in signs:
            continue
        by_sign[row["sign_id"]].append(row)

    selected: list[dict] = []
    for sign_id in sorted(by_sign)[:max_signs]:
        rows = sorted(by_sign[sign_id], key=lambda r: (r.get("signer_id", ""), r.get("path", "")))
        selected.extend(rows[:clips_per_sign])
    return selected


def _draw_clip_cell(draw: ImageDraw.ImageDraw, sheet: Image.Image, x: int, y: int, row: dict,
                    processed: np.ndarray, raw_images: list[Image.Image] | None, frame_count: int) -> None:
    title_font = _font(13)
    small_font = _font(11)
    label = f"{row['sign_id']} / {row.get('signer_id', 'unknown')} / {Path(row['path']).name}"
    draw.text((x, y), label[:80], fill=INK, font=title_font)
    draw.text((x, y + 16), "raw crop box -> imported model frame" if raw_images else "imported model frames", fill=MUTED, font=small_font)

    proc_idx = _resample_indices(processed.shape[0], frame_count)
    for col, idx in enumerate(proc_idx):
        cx = x + col * (CELL_W + PAD)
        cy = y + LABEL_H
        if raw_images:
            raw = raw_images[col]
            raw_box = _letterbox(raw, (CELL_W, CELL_H_RAW))
            r = _crop_rect_in_letterbox(raw, (CELL_W, CELL_H_RAW))
            raw_draw = ImageDraw.Draw(raw_box)
            raw_draw.rectangle(r, outline=ACCENT, width=3)
            raw_draw.text((4, 4), f"raw {idx + 1}", fill=(255, 255, 255), font=small_font)
            sheet.paste(raw_box, (cx, cy))
            cy += CELL_H_RAW + 4
        proc = _letterbox(_to_image(processed[idx]), (CELL_W, CELL_H_PROCESSED))
        proc_draw = ImageDraw.Draw(proc)
        proc_draw.text((4, 4), f"npz {idx + 1}", fill=(255, 255, 255), font=small_font)
        sheet.paste(proc, (cx, cy))


def render_sheet(rows: list[dict], out_path: Path, video_dir: Path, frames_per_clip: int) -> tuple[int, int]:
    raw_index = _raw_video_index(video_dir)
    row_h = LABEL_H + CELL_H_PROCESSED + 2 * PAD
    if raw_index:
        row_h += CELL_H_RAW + 4
    width = PAD + frames_per_clip * (CELL_W + PAD)
    height = PAD + max(1, len(rows)) * row_h
    sheet = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(sheet)

    raw_matches = 0
    for i, row in enumerate(rows):
        y = PAD + i * row_h
        clip_path = Path(row["path"])
        if not clip_path.is_absolute():
            clip_path = ROOT / clip_path
        processed = load_clip(clip_path)

        raw_images = None
        raw_path = _source_video_from_npz(clip_path)
        if raw_path is None:
            clip_num = _clip_num_from_npz(clip_path)
            if clip_num:
                raw_path = raw_index.get((row["sign_id"], row.get("signer_id", ""), clip_num))
        if raw_path:
            try:
                raw_images = _read_video_frames(raw_path, frames_per_clip)
                raw_matches += 1
            except Exception as exc:
                print(f"WARNING: could not read raw video {raw_path}: {exc}", file=sys.stderr)
        _draw_clip_cell(draw, sheet, PAD, y, row, processed, raw_images, frames_per_clip)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return len(rows), raw_matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "ml" / "data" / "manifest.json")
    parser.add_argument("--video-dir", type=Path, default=ROOT / "ml" / "data" / "incoming")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "ml" / "reports" / "contact_sheets")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--signs", default="", help="Comma-separated sign IDs. Defaults to first --max-signs signs.")
    parser.add_argument("--max-signs", type=int, default=8)
    parser.add_argument("--clips-per-sign", type=int, default=2)
    parser.add_argument("--frames-per-clip", type=int, default=6)
    parser.add_argument("--filename", default="", help="Optional output filename. Defaults to contact_sheet_<split>.jpg.")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    signs = {s.strip() for s in args.signs.split(",") if s.strip()} or None
    rows = _select_rows(manifest, args.split, signs, args.max_signs, args.clips_per_sign)
    if not rows:
        raise SystemExit(f"No clips matched split={args.split!r} signs={sorted(signs) if signs else 'auto'}")

    filename = args.filename or f"contact_sheet_{args.split}.jpg"
    out_path = args.out_dir / filename
    count, raw_matches = render_sheet(rows, out_path, args.video_dir, args.frames_per_clip)
    print(f"Wrote {out_path} ({count} clips, {raw_matches} with matching raw videos)")
    if raw_matches == 0:
        print("No matching raw videos found; sheet shows imported .npz model frames only.")


if __name__ == "__main__":
    main()
