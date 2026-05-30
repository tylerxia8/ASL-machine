"""Fetch a small WLASL subset by gloss into ml/data/incoming_online_wlasl.

WLASL index:
https://github.com/dxli94/WLASL

This fetcher is intentionally conservative: it downloads only direct video URLs
from the public JSON index. YouTube entries are skipped unless a future pass
adds an explicit yt-dlp dependency and clipping workflow.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "ml" / "data" / "incoming_online_wlasl"
INDEX_URL = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv", ".avi")


def _sign_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _load_index(index_path: Path | None) -> list[dict]:
    if index_path and index_path.exists():
        return json.loads(index_path.read_text(encoding="utf-8"))
    with urllib.request.urlopen(INDEX_URL, timeout=60) as res:
        return json.loads(res.read().decode("utf-8"))


def _is_direct_video(url: str) -> bool:
    clean = url.split("?", 1)[0].lower()
    return clean.endswith(VIDEO_EXTS)


def _download(url: str, out_path: Path) -> bool:
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    try:
        with requests.get(url, stream=True, timeout=60, allow_redirects=True) as res:
            res.raise_for_status()
            ctype = res.headers.get("Content-Type", "").lower()
            if "text/html" in ctype:
                print(f"  skip html response: {url}")
                return False
            tmp = out_path.with_suffix(out_path.suffix + ".tmp")
            with open(tmp, "wb") as f:
                for chunk in res.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            if tmp.stat().st_size == 0:
                tmp.unlink(missing_ok=True)
                return False
            tmp.replace(out_path)
            return True
    except requests.RequestException as e:
        print(f"  failed {url}: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signs", required=True, help="Comma-separated glosses/sign IDs to fetch.")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--index-json", help="Optional local WLASL_v0.3.json path.")
    parser.add_argument("--clips-per-sign", type=int, default=20)
    args = parser.parse_args()

    signs = [_sign_id(s.strip()) for s in args.signs.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    index = _load_index(Path(args.index_json) if args.index_json else None)
    by_gloss = {_sign_id(row.get("gloss", "")): row for row in index}

    total = 0
    for sign in signs:
        row = by_gloss.get(sign)
        if not row:
            print(f"{sign}: no WLASL entry")
            continue
        written = 0
        skipped_youtube = 0
        skipped_nonvideo = 0
        print(f"{sign}: {len(row.get('instances', []))} indexed instance(s)")
        for inst in row.get("instances", []):
            if written >= args.clips_per_sign:
                break
            url = inst.get("url") or ""
            video_id = str(inst.get("video_id") or "")
            if "youtube.com" in url or "youtu.be" in url:
                skipped_youtube += 1
                continue
            if not _is_direct_video(url):
                skipped_nonvideo += 1
                continue
            suffix = Path(url.split("?", 1)[0]).suffix.lower() or ".mp4"
            signer = f"signer_wlasl{re.sub(r'[^a-zA-Z0-9]+', '', video_id).lower()}"
            out_path = out_dir / f"{sign}_{signer}_{written + 1:04d}{suffix}"
            if _download(url, out_path):
                written += 1
                total += 1
                print(f"  {video_id} -> {out_path.name}")
        print(
            f"  wrote {written}; skipped_youtube={skipped_youtube}; "
            f"skipped_nonvideo={skipped_nonvideo}"
        )

    print(f"\nDone: {total} WLASL direct video(s) -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
