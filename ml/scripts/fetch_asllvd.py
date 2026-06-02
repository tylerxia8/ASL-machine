"""Fetch small ASLLVD citation-form clips by gloss.

ASLLVD source:
https://www.bu.edu/asllrp/av/dai-asllvd.html

The spreadsheet provides scene, start frame, and end frame annotations. This
script downloads only the scene videos needed for the requested exact glosses,
then trims them locally with OpenCV. Raw scenes and trimmed clips live under
ignored ml/data folders and must not be committed.
"""
from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import cv2
import requests

ROOT = Path(__file__).resolve().parents[2]
INDEX_URL = "https://www.bu.edu/asllrp/dai-asllvd-BU_glossing_with_variations_HS_information-extended-urls-RU.xlsx"
SCENE_BASE = "http://csr.bu.edu/ftp/asl/asllvd/asl-data2/quicktime"
INDEX_PATH = ROOT / "ml" / "data" / "external_indexes" / "asllvd_extended_urls.xlsx"
SCENE_CACHE = ROOT / "ml" / "data" / "external_indexes" / "asllvd_scenes"
OUT_DIR = ROOT / "ml" / "data" / "incoming_online_asllvd"
NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _sign_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _colnum(ref: str) -> int:
    letters = "".join(ch for ch in ref if ch.isalpha())
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch.upper()) - 64
    return n - 1


def _download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=120) as res:
        res.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in res.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(path)


def _load_rows(index_path: Path) -> list[dict[str, str]]:
    _download(INDEX_URL, index_path)
    with ZipFile(index_path) as z:
        shared_root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        shared = [
            "".join((t.text or "") for t in si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"))
            for si in shared_root.findall("a:si", NS)
        ]
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        raw_rows: list[dict[int, str]] = []
        for row in sheet.findall("a:sheetData/a:row", NS):
            vals: dict[int, str] = {}
            for cell in row.findall("a:c", NS):
                v = cell.find("a:v", NS)
                if v is None:
                    continue
                val = v.text or ""
                if cell.attrib.get("t") == "s" and val:
                    val = shared[int(val)]
                vals[_colnum(cell.attrib["r"])] = val
            raw_rows.append(vals)

    headers = raw_rows[0]
    rows: list[dict[str, str]] = []
    for raw in raw_rows[1:]:
        row = {headers.get(k, str(k)): v for k, v in raw.items()}
        if row.get("Main New Gloss") == "============":
            continue
        rows.append(row)
    return rows


def _scene_url(session: str, scene: str) -> str:
    return f"{SCENE_BASE}/{session}/scene{scene}-camera1.mov"


def _trim_clip(scene_path: Path, out_path: Path, start_frame: int, end_frame: int, pad_frames: int) -> bool:
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    cap = cv2.VideoCapture(str(scene_path))
    if not cap.isOpened():
        print(f"  failed to open {scene_path}")
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    start = max(0, start_frame - pad_frames)
    end = min(total - 1, end_frame + pad_frames) if total else end_frame + pad_frames
    if end <= start or width <= 0 or height <= 0:
        cap.release()
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    written = 0
    for _ in range(start, end + 1):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    writer.release()
    cap.release()
    if written == 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(out_path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signs", required=True, help="Comma-separated sign ids/glosses to fetch.")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--clips-per-sign", type=int, default=20)
    parser.add_argument("--index-xlsx", default=str(INDEX_PATH))
    parser.add_argument("--scene-cache", default=str(SCENE_CACHE))
    parser.add_argument("--pad-frames", type=int, default=6)
    args = parser.parse_args()

    signs = [_sign_id(s.strip()) for s in args.signs.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    index_path = Path(args.index_xlsx)
    if not index_path.is_absolute():
        index_path = ROOT / index_path
    scene_cache = Path(args.scene_cache)
    if not scene_cache.is_absolute():
        scene_cache = ROOT / scene_cache

    rows = _load_rows(index_path)
    total = 0
    for sign in signs:
        matches = [
            row for row in rows
            if _sign_id(row.get("Main New Gloss", "")) == sign or _sign_id(row.get("Gloss Variant", "")) == sign
        ][: args.clips_per_sign]
        print(f"{sign}: {len(matches)} exact ASLLVD row(s)")
        written = 0
        for row in matches:
            session = row.get("Session", "")
            scene = row.get("Scene", "")
            if not session or not scene:
                continue
            try:
                start = int(float(row.get("Start", "")))
                end = int(float(row.get("End", "")))
            except ValueError:
                continue
            signer = f"signer_asllvd{_sign_id(row.get('Consultant', 'unknown'))}"
            scene_path = scene_cache / session / f"scene{scene}-camera1.mov"
            print(f"  scene {session}/{scene} {start}-{end}")
            _download(_scene_url(session, scene), scene_path)
            out_path = out_dir / f"{sign}_{signer}_{written + 1:04d}.mp4"
            if _trim_clip(scene_path, out_path, start, end, args.pad_frames):
                written += 1
                total += 1
                print(f"    -> {out_path.name}")
        print(f"  wrote {written}")
    print(f"\nDone: {total} ASLLVD clip(s) -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
