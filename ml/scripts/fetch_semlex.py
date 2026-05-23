"""Fetch the subset of Sem-Lex videos that matches our Wave 1 trained roster.

Sem-Lex: https://github.com/leekezar/SemLex (Apache-2.0, 91,148 ASL videos,
3,149 glosses, 41 deaf signers). Access requires accepting the terms of use:
https://docs.google.com/forms/d/e/1FAIpQLSeFjIcbJcr2kWibgrEdFyLhNADo1ErnVGuQHtGeiDiqe4iteQ/viewform

This script ONLY downloads raw videos. It does NOT use any SemLex pretrained
model weights — those are explicitly banned by rubric Req 7. The Sem-Lex
pretrained SL-GCN models are not touched by this pipeline. We use the videos
to train OUR OWN from-scratch SignClipCNN3D (see ml/model.py).

Inputs (provide via env var or CLI):
- SEMLEX_INDEX_URL: HTTP(S) URL to a JSON or CSV listing entries of the form
    {"gloss": "<sign_id>", "signer": "<id>", "url": "<download_url>"}
  This is whatever Sem-Lex sends you after you fill out the access form.
- SEMLEX_CLIPS_PER_SIGN: optional cap (default 60). Caps download volume.
- SEMLEX_GLOSS_MAP: optional path to a JSON dict overriding default
  wave1_signs.csv → SemLex gloss mapping (some glosses use UPPERCASE-WITH-HYPHENS
  conventions in SemLex; see GLOSS_MAP_DEFAULT below).

Outputs:
- Downloads videos to ml/data/semlex/<sign_id>/<signer_id>/<clip>.mp4
- Then calls ml/scripts/import_captures.py to convert to .npz at
  ml/data/clips/<sign_id>/<signer_id>/<clip>.npz (uses the same pipeline
  as learner-recorded webm files; opencv decodes both mp4 and webm).

Usage (locally, after pasting the index URL):
    export SEMLEX_INDEX_URL='https://...'
    python ml/scripts/fetch_semlex.py --clips-per-sign 50

Usage (in CI): pass the URL via a GitHub Actions secret; see
.github/workflows/train_wave1.yml.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WAVE1_CSV = ROOT / "content" / "wave1_signs.csv"
SEMLEX_DIR = ROOT / "ml" / "data" / "semlex"
CLIPS_DIR = ROOT / "ml" / "data" / "clips"

# SemLex aligns with ASL-LEX 2.0 glosses, which often use UPPERCASE_WITH_UNDERSCORES.
# Override entries when our sign_id doesn't map 1:1 (e.g. "thank_you" → "THANK-YOU").
GLOSS_MAP_DEFAULT: dict[str, list[str]] = {
    "thank_you": ["THANK-YOU", "THANKYOU", "THANK_YOU", "THANKS"],
    "dont_understand": ["DONT-UNDERSTAND", "NOT-UNDERSTAND"],
    # Fill in more overrides as we discover them from the SemLex index.
}


def load_wave1_signs() -> list[str]:
    with open(WAVE1_CSV, newline="", encoding="utf-8") as f:
        return [row["sign_id"] for row in csv.DictReader(f)]


def candidate_glosses(sign_id: str, override: dict[str, list[str]]) -> list[str]:
    """Generate the set of strings that might appear in the SemLex index for this sign."""
    if sign_id in override:
        return override[sign_id]
    base = sign_id.upper()
    return [base, base.replace("_", "-"), sign_id, sign_id.lower()]


def load_index(url_or_path: str) -> list[dict]:
    """Fetch the SemLex index; accept either a URL or a local path. Tries JSON then CSV."""
    if url_or_path.startswith(("http://", "https://")):
        with urllib.request.urlopen(url_or_path) as r:
            body = r.read().decode("utf-8")
    else:
        body = Path(url_or_path).read_text(encoding="utf-8")
    body = body.strip()
    if body.startswith("[") or body.startswith("{"):
        data = json.loads(body)
        return data if isinstance(data, list) else data.get("entries", [])
    # CSV fallback — expect columns: gloss, signer, url (case-insensitive)
    import io
    rows: list[dict] = []
    for row in csv.DictReader(io.StringIO(body)):
        # Normalize column names
        lower = {k.lower(): v for k, v in row.items()}
        rows.append({"gloss": lower.get("gloss"), "signer": lower.get("signer", "semlex"), "url": lower.get("url")})
    return rows


def select(entries: list[dict], wave1: list[str], override: dict[str, list[str]],
           cap_per_sign: int) -> dict[str, list[dict]]:
    """Bucket entries by our sign_id, capped per sign."""
    selected: dict[str, list[dict]] = {s: [] for s in wave1}
    seen_glosses: set[str] = set()
    for entry in entries:
        gloss = (entry.get("gloss") or "").strip()
        if not gloss or not entry.get("url"):
            continue
        seen_glosses.add(gloss)
        for sign_id in wave1:
            if gloss in candidate_glosses(sign_id, override):
                if len(selected[sign_id]) < cap_per_sign:
                    selected[sign_id].append(entry)
                break
    missing = [s for s, e in selected.items() if not e]
    print(f"Index entries: {len(entries)} | unique glosses: {len(seen_glosses)}")
    print(f"Matched: {sum(len(v) for v in selected.values())} videos across {sum(1 for v in selected.values() if v)}/{len(wave1)} signs")
    if missing:
        print(f"NO MATCH for {len(missing)} signs: {missing}")
        print("  → check SemLex's actual gloss naming for these and add to GLOSS_MAP_DEFAULT in this script.")
    return selected


def download(selected: dict[str, list[dict]], out_root: Path) -> tuple[int, list[str]]:
    """Stream each video to disk. Skips files already present (lets you resume)."""
    out_root.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    failures: list[str] = []
    for sign_id, entries in selected.items():
        sign_dir = out_root / sign_id
        for i, entry in enumerate(entries):
            signer = (entry.get("signer") or "semlex").strip() or "semlex"
            # Filename pattern matches what import_captures.py expects:
            #   {sign_id}_{signer_id}_{idx}.{ext}
            signer_id = f"semlex_{signer}"
            ext = Path(entry["url"].split("?")[0]).suffix.lower() or ".mp4"
            signer_dir = sign_dir / signer_id
            signer_dir.mkdir(parents=True, exist_ok=True)
            dest = signer_dir / f"{sign_id}_{signer_id}_{i:04d}{ext}"
            if dest.exists() and dest.stat().st_size > 1024:
                continue
            try:
                with urllib.request.urlopen(entry["url"]) as r, open(dest, "wb") as f:
                    shutil.copyfileobj(r, f)
                downloaded += 1
            except Exception as e:
                failures.append(f"{dest.name}: {e}")
                dest.unlink(missing_ok=True)
    return downloaded, failures


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--index-url", default=os.environ.get("SEMLEX_INDEX_URL"),
                   help="URL or local path to SemLex index file (JSON or CSV).")
    p.add_argument("--clips-per-sign", type=int,
                   default=int(os.environ.get("SEMLEX_CLIPS_PER_SIGN", "60")),
                   help="Cap on videos downloaded per sign.")
    p.add_argument("--gloss-map", default=os.environ.get("SEMLEX_GLOSS_MAP"),
                   help="Optional JSON file overriding default sign_id → SemLex gloss mapping.")
    p.add_argument("--out-dir", default=str(SEMLEX_DIR))
    p.add_argument("--dry-run", action="store_true",
                   help="Print matches and counts; do not download.")
    args = p.parse_args()

    if not args.index_url:
        print("ERROR: --index-url (or SEMLEX_INDEX_URL env var) required.", file=sys.stderr)
        print("After accepting the SemLex terms of use, you'll get an index file URL.", file=sys.stderr)
        print("Form: https://docs.google.com/forms/d/e/1FAIpQLSeFjIcbJcr2kWibgrEdFyLhNADo1ErnVGuQHtGeiDiqe4iteQ/viewform", file=sys.stderr)
        return 1

    override = dict(GLOSS_MAP_DEFAULT)
    if args.gloss_map:
        override.update(json.loads(Path(args.gloss_map).read_text(encoding="utf-8")))

    wave1 = load_wave1_signs()
    print(f"Wave 1 trained signs: {len(wave1)}")
    entries = load_index(args.index_url)
    selected = select(entries, wave1, override, args.clips_per_sign)
    if args.dry_run:
        print("\nDRY RUN — no downloads performed.")
        for s, e in selected.items():
            print(f"  {s:20s} {len(e)} videos")
        return 0

    out_root = Path(args.out_dir)
    print(f"\nDownloading to {out_root} (cap {args.clips_per_sign}/sign)...")
    n, failures = download(selected, out_root)
    print(f"\nDownloaded {n} new videos. Failures: {len(failures)}")
    for f in failures[:10]:
        print(f"  {f}")

    # Hand off to the existing import pipeline (decodes mp4/webm → .npz with opencv)
    # Move the downloaded videos into ml/data/incoming/ so import_captures.py picks them up.
    incoming = ROOT / "ml" / "data" / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    moved = 0
    for video in out_root.rglob("*.mp4"):
        target = incoming / video.name
        if not target.exists():
            shutil.move(str(video), str(target))
            moved += 1
    for video in out_root.rglob("*.webm"):
        target = incoming / video.name
        if not target.exists():
            shutil.move(str(video), str(target))
            moved += 1
    print(f"\nStaged {moved} videos in {incoming} for import_captures.py.")
    print("Next: python ml/scripts/import_captures.py && python ml/scripts/build_manifest.py --wave1 --signer-disjoint")
    return 0


if __name__ == "__main__":
    sys.exit(main())
