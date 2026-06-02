"""Fetch small targeted PopSign ASL supplements by sign.

PopSign source:
https://signdata.cc.gatech.edu/view/datasets/popsign_v1_0/

The full dataset is large, so this script streams per-sign tar files and stops
after a small clip cap. Use it as a targeted learner-style supplement, not as a
bulk mirror.
"""
from __future__ import annotations

import argparse
import re
import shutil
import tarfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "ml" / "data" / "incoming_online_popsign"
BASE_URL = "https://signdata.cc.gatech.edu/data"
VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv", ".avi")

SIGN_ALIASES = {
    "goodbye": "bye",
    "thank_you": "thankyou",
}


def _sign_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _signer_id(member_name: str) -> str:
    filename = Path(member_name).name
    match = re.search(r"(?:gtsignstudy)?4a\.(\d+)", filename)
    if match:
        return f"signer_popsign{match.group(1)}"
    match = re.search(r"\.(\d+)-", filename)
    if match:
        return f"signer_popsign{match.group(1)}"
    return "signer_popsignunknown"


def _download_sign(
    *,
    dataset: str,
    category: str,
    split: str,
    sign_id: str,
    popsign_sign: str,
    out_dir: Path,
    clips_per_sign: int,
    max_clip_bytes: int,
    start_index: int,
) -> int:
    url = f"{BASE_URL}/{dataset}/{category}/{split}/{popsign_sign}.tar"
    print(f"{sign_id}: streaming {url}")
    written = 0
    skipped_large = 0
    skipped_nonvideo = 0
    try:
        with requests.get(url, stream=True, timeout=180) as res:
            if res.status_code == 404:
                print(f"  no PopSign tar for {popsign_sign!r} in {split}")
                return 0
            res.raise_for_status()
            res.raw.decode_content = True
            with tarfile.open(fileobj=res.raw, mode="r|") as tar:
                for member in tar:
                    if written >= clips_per_sign:
                        break
                    if not member.isfile() or not member.name.lower().endswith(VIDEO_EXTS):
                        skipped_nonvideo += 1
                        continue
                    if member.size > max_clip_bytes:
                        skipped_large += 1
                        continue
                    source = tar.extractfile(member)
                    if source is None:
                        continue
                    suffix = Path(member.name).suffix.lower() or ".mp4"
                    signer = _signer_id(member.name)
                    out_path = out_dir / f"{sign_id}_{signer}_{start_index + written + 1:04d}{suffix}"
                    if out_path.exists() and out_path.stat().st_size > 0:
                        written += 1
                        continue
                    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
                    with open(tmp, "wb") as f:
                        shutil.copyfileobj(source, f)
                    if tmp.stat().st_size == 0:
                        tmp.unlink(missing_ok=True)
                        continue
                    tmp.replace(out_path)
                    written += 1
                    print(f"  {member.name} -> {out_path.name}")
    except (requests.RequestException, tarfile.TarError, OSError) as exc:
        print(f"  failed {url}: {exc}")
        return written

    print(
        f"  wrote {written}; skipped_large={skipped_large}; "
        f"skipped_nonvideo={skipped_nonvideo}"
    )
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signs", required=True, help="Comma-separated Wave 1 sign IDs to fetch.")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--dataset", default="popsign_v1_0")
    parser.add_argument("--category", default="game")
    parser.add_argument(
        "--splits",
        default="test",
        help="Comma-separated PopSign tar splits to stream. Default uses the smaller test tars.",
    )
    parser.add_argument("--clips-per-sign", type=int, default=8)
    parser.add_argument(
        "--max-clip-mb",
        type=float,
        default=30.0,
        help="Skip individual clips larger than this many MB.",
    )
    args = parser.parse_args()

    signs = [_sign_id(s.strip()) for s in args.signs.split(",") if s.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    max_clip_bytes = int(args.max_clip_mb * 1024 * 1024)

    total = 0
    for sign in signs:
        popsign_sign = SIGN_ALIASES.get(sign, sign)
        written_for_sign = 0
        for split in splits:
            if written_for_sign >= args.clips_per_sign:
                break
            written = _download_sign(
                dataset=args.dataset,
                category=args.category,
                split=split,
                sign_id=sign,
                popsign_sign=popsign_sign,
                out_dir=out_dir,
                clips_per_sign=args.clips_per_sign - written_for_sign,
                max_clip_bytes=max_clip_bytes,
                start_index=written_for_sign,
            )
            written_for_sign += written
            total += written
        if written_for_sign == 0 and popsign_sign != sign:
            print(f"  alias used: {sign} -> {popsign_sign}")

    print(f"\nDone: {total} PopSign clip(s) -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
