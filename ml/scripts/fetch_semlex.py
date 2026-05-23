"""Fetch the subset of Sem-Lex videos matching our Wave 1 trained roster.

Sem-Lex: https://github.com/leekezar/SemLex (Apache-2.0, 91,148 ASL videos,
3,149 glosses, 41 deaf signers). Access requires accepting the terms of use:
https://docs.google.com/forms/d/e/1FAIpQLSeFjIcbJcr2kWibgrEdFyLhNADo1ErnVGuQHtGeiDiqe4iteQ/viewform

After acceptance Sem-Lex provides Google Drive links for SEVEN files:
- semlex_metadata.csv       — index mapping filename → gloss / signer / split
- train.tar.gz              — train-split videos
- val.tar.gz                — val-split videos
- test.tar.gz               — test-split videos
- sem-lex-train-poses.tar.gz   <-- NOT DOWNLOADED. These contain precomputed
- sem-lex-val-poses.tar.gz     <-- pose features from a pretrained landmark
- sem-lex-test-poses.tar.gz    <-- detector. Using them would violate rubric
                                   Req 7 (no pretrained landmark detectors).

This script downloads only the 4 video / metadata files (skipping the 3 pose
archives entirely — never referenced anywhere in the code path), filters to
our 25-sign Wave 1 roster via the metadata CSV, and stages selected videos
into ml/data/incoming/ for the existing import_captures.py pipeline.

The tarballs are streamed: we read the archive sequentially and only extract
videos that match our roster. Peak disk: roughly the size of all selected
videos (~1–2 GB), not the full archive (~50–100 GB).

Inputs (provide via env var, JSON-encoded):
    SEMLEX_DRIVE_FILES='{
        "metadata": "<drive_file_id>",
        "train":    "<drive_file_id>",
        "val":      "<drive_file_id>",
        "test":     "<drive_file_id>"
    }'

`metadata` is required. Any of `train` / `val` / `test` are optional — only
the listed tarballs are downloaded. To keep network and disk small for a
shakedown run, you can supply only `val` (smallest split).

Optional:
    SEMLEX_CLIPS_PER_SIGN  cap on extracted videos per sign (default 60)
    SEMLEX_GLOSS_MAP       JSON file overriding sign_id → Sem-Lex gloss
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import tarfile
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
WAVE1_CSV = ROOT / "content" / "wave1_signs.csv"
SEMLEX_DIR = ROOT / "ml" / "data" / "semlex"
INCOMING_DIR = ROOT / "ml" / "data" / "incoming"

# Sem-Lex aligns with ASL-LEX 2.0 / ASL SignBank glosses. Common conventions:
#   UPPERCASE_WITH_UNDERSCORES, or UPPERCASE-WITH-HYPHENS, or lowercase.
# When our internal sign_id and Sem-Lex's gloss differ (e.g. our "thank_you"
# vs Sem-Lex's "THANK-YOU"), list the Sem-Lex variants here.
GLOSS_MAP_DEFAULT: dict[str, list[str]] = {
    "thank_you": ["THANK-YOU", "THANK_YOU", "THANKYOU", "THANKS"],
    "dont_understand": ["DONT-UNDERSTAND", "NOT-UNDERSTAND"],
    "sign_language": ["SIGN-LANGUAGE", "SIGN_LANGUAGE", "SIGN"],
}

# Common column names we'll probe for in the metadata CSV. Sem-Lex may use any.
FILENAME_COLS = ("filename", "file", "video", "video_file", "clip", "clip_id", "name", "id")
GLOSS_COLS = ("gloss", "label", "sign", "class", "lemma", "entry_id", "EntryID")
SIGNER_COLS = ("signer", "signer_id", "participant", "participant_id", "subject", "subject_id")
SPLIT_COLS = ("split", "set", "partition", "fold")


def load_wave1_signs() -> list[str]:
    with open(WAVE1_CSV, newline="", encoding="utf-8") as f:
        return [row["sign_id"] for row in csv.DictReader(f)]


def candidate_glosses(sign_id: str, override: dict[str, list[str]]) -> set[str]:
    """Strings that might appear in the Sem-Lex metadata for this sign."""
    if sign_id in override:
        return {g.upper() for g in override[sign_id]}
    base = sign_id.upper()
    return {base, base.replace("_", "-"), base.replace("_", ""), sign_id.upper()}


def pick_column(fieldnames: Iterable[str], candidates: Iterable[str]) -> str | None:
    lower_to_orig = {f.lower(): f for f in fieldnames}
    for c in candidates:
        if c.lower() in lower_to_orig:
            return lower_to_orig[c.lower()]
    return None


def _gdown_to(file_id: str, dest: Path) -> Path:
    """Download a Google Drive file by ID. Resumes if dest is partially complete."""
    import gdown
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading {file_id} → {dest}")
    gdown.download(url, str(dest), quiet=False, resume=True)
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"gdown produced empty/missing file at {dest}")
    print(f"  done: {dest.stat().st_size / 1e6:.1f} MB")
    return dest


def load_metadata(csv_path: Path, wave1: list[str], override: dict[str, list[str]]
                  ) -> dict[str, tuple[str, str, str]]:
    """Return {basename_in_archive: (sign_id, signer_id, split)} for matching videos."""
    wanted: dict[str, set[str]] = {s: candidate_glosses(s, override) for s in wave1}

    matches: dict[str, tuple[str, str, str]] = {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fname_col = pick_column(reader.fieldnames or [], FILENAME_COLS)
        gloss_col = pick_column(reader.fieldnames or [], GLOSS_COLS)
        signer_col = pick_column(reader.fieldnames or [], SIGNER_COLS)
        split_col = pick_column(reader.fieldnames or [], SPLIT_COLS)
        if not fname_col or not gloss_col:
            raise SystemExit(
                f"Could not find filename + gloss columns in {csv_path}.\n"
                f"Available columns: {reader.fieldnames}\n"
                f"Probed filename names: {FILENAME_COLS}\n"
                f"Probed gloss names: {GLOSS_COLS}\n"
                f"Either rename the column in the CSV or add to the *_COLS lists in fetch_semlex.py."
            )
        print(f"Metadata columns → filename={fname_col!r} gloss={gloss_col!r} signer={signer_col!r} split={split_col!r}")

        unique_glosses: set[str] = set()
        for row in reader:
            gloss_raw = (row.get(gloss_col) or "").strip()
            if not gloss_raw:
                continue
            unique_glosses.add(gloss_raw)
            gloss_upper = gloss_raw.upper()
            # Find which sign_id this row maps to (if any).
            matched: str | None = None
            for sign_id, variants in wanted.items():
                if gloss_upper in variants:
                    matched = sign_id
                    break
            if not matched:
                continue
            fname = (row.get(fname_col) or "").strip()
            if not fname:
                continue
            # Normalize to just the basename — archive members may have prefix dirs.
            basename = Path(fname).name
            signer_raw = (row.get(signer_col) or "unknown").strip() if signer_col else "unknown"
            signer_id = f"semlex_{re.sub(r'[^A-Za-z0-9]+', '_', signer_raw).lower()}".strip("_") or "semlex_unknown"
            split = (row.get(split_col) or "unknown").strip() if split_col else "unknown"
            matches[basename] = (matched, signer_id, split)

    print(f"Metadata rows: matched {len(matches)} videos covering "
          f"{len({m[0] for m in matches.values()})}/{len(wave1)} signs "
          f"(out of {len(unique_glosses)} unique Sem-Lex glosses)")
    return matches


def stream_extract(tar_path: Path, wanted: dict[str, tuple[str, str, str]],
                   cap_per_sign: int, out_dir: Path) -> int:
    """Sequentially stream tar.gz; extract only members whose basename matches."""
    counts: dict[str, int] = {}
    extracted = 0
    print(f"Streaming {tar_path.name} → extracting only matched videos...")
    # mode='r|gz' is the streaming reader (no seek).
    with tarfile.open(tar_path, mode="r|gz") as tar:
        for i, member in enumerate(tar, start=1):
            if i % 5000 == 0:
                print(f"  scanned {i} members ({extracted} extracted so far)")
            if not member.isfile():
                continue
            basename = Path(member.name).name
            meta = wanted.get(basename)
            if not meta:
                continue
            sign_id, signer_id, _split = meta
            if counts.get(sign_id, 0) >= cap_per_sign:
                continue
            ext = Path(basename).suffix.lower() or ".mp4"
            counts[sign_id] = counts.get(sign_id, 0) + 1
            target = out_dir / f"{sign_id}_{signer_id}_semlex_{counts[sign_id]:04d}{ext}"
            with tar.extractfile(member) as src, open(target, "wb") as dst:
                if src is None:
                    continue
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            extracted += 1
    print(f"  done: {extracted} videos extracted from {tar_path.name}")
    return extracted


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--drive-files-json", default=os.environ.get("SEMLEX_DRIVE_FILES"),
                   help='JSON dict: {"metadata":"<id>","train":"<id>","val":"<id>","test":"<id>"}. '
                        'Only `metadata` is required; any of train/val/test are optional.')
    p.add_argument("--clips-per-sign", type=int,
                   default=int(os.environ.get("SEMLEX_CLIPS_PER_SIGN", "60")))
    p.add_argument("--gloss-map", default=os.environ.get("SEMLEX_GLOSS_MAP"))
    p.add_argument("--work-dir", default=str(SEMLEX_DIR),
                   help="Where to stash downloaded tarballs (temporary; can be cached in CI).")
    p.add_argument("--out-dir", default=str(INCOMING_DIR),
                   help="Where extracted videos land for the import pipeline.")
    p.add_argument("--keep-tarballs", action="store_true",
                   help="Keep downloaded .tar.gz files (default: delete after extraction to save disk).")
    args = p.parse_args()

    if not args.drive_files_json:
        print("ERROR: SEMLEX_DRIVE_FILES not set.", file=sys.stderr)
        print("Expected JSON dict mapping role → Google Drive file ID:", file=sys.stderr)
        print('  {"metadata":"...","train":"...","val":"...","test":"..."}', file=sys.stderr)
        return 1

    try:
        drive_files: dict[str, str] = json.loads(args.drive_files_json)
    except json.JSONDecodeError as e:
        print(f"ERROR: SEMLEX_DRIVE_FILES is not valid JSON: {e}", file=sys.stderr)
        return 1

    if "metadata" not in drive_files:
        print("ERROR: SEMLEX_DRIVE_FILES must include a 'metadata' key (semlex_metadata.csv).", file=sys.stderr)
        return 1

    work_dir = Path(args.work_dir)
    out_dir = Path(args.out_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    override = dict(GLOSS_MAP_DEFAULT)
    if args.gloss_map:
        override.update(json.loads(Path(args.gloss_map).read_text(encoding="utf-8")))

    wave1 = load_wave1_signs()
    print(f"Wave 1 trained signs: {len(wave1)}")

    # 1. Pull metadata, identify which video filenames we want.
    metadata_csv = _gdown_to(drive_files["metadata"], work_dir / "semlex_metadata.csv")
    wanted = load_metadata(metadata_csv, wave1, override)
    if not wanted:
        print("FATAL: 0 videos matched the wave1 roster. Check GLOSS_MAP_DEFAULT and the CSV's column naming.", file=sys.stderr)
        return 2

    total_extracted = 0
    for role in ("train", "val", "test"):
        if role not in drive_files:
            print(f"Skipping {role} (not in SEMLEX_DRIVE_FILES).")
            continue
        tar_path = _gdown_to(drive_files[role], work_dir / f"{role}.tar.gz")
        n = stream_extract(tar_path, wanted, args.clips_per_sign, out_dir)
        total_extracted += n
        if not args.keep_tarballs:
            tar_path.unlink(missing_ok=True)
            print(f"  removed {tar_path.name} ({tar_path.name} no longer on disk; saves space for next archive)")

    print(f"\nTotal extracted: {total_extracted} videos → {out_dir}")
    print("Next: python ml/scripts/import_captures.py && python ml/scripts/build_manifest.py --wave1 --signer-disjoint")
    return 0


if __name__ == "__main__":
    sys.exit(main())
