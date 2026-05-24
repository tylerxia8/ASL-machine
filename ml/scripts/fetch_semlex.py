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

# Sem-Lex labels (verified from semlex_metadata.csv) are lowercase with
# underscores: "hello", "eat_1", "thank_you", "bye", etc. Sem-Lex frequently
# disambiguates sign variants with numeric suffixes (eat_1, deaf_1, what_1).
# Our matcher includes "{sign_id}_<digit>" patterns automatically; the override
# map below handles cases where Sem-Lex uses a different lemma entirely
# (e.g. our "goodbye" vs Sem-Lex's "bye").
GLOSS_MAP_DEFAULT: dict[str, list[str]] = {
    "thank_you": ["thank_you", "thanks", "thank-you", "thankyou"],
    "dont_understand": ["dont_understand", "not_understand", "dont-understand"],
    "sign_language": ["sign_language", "sign-language", "sign"],
    # Sem-Lex has 0 plain "goodbye" in asllex but 17 "bye". Include all variants.
    "goodbye": ["goodbye", "bye", "good_bye", "farewell"],
}

# Wave-1 signs with near-zero Sem-Lex coverage. The model still has output
# heads for these (loaded from wave1_signs.csv), but training signal is weak
# and per-class accuracy will be very low. Document in VALIDATION_REPORT.md.
KNOWN_LOW_COVERAGE: set[str] = {"five"}

# Sem-Lex `label_type` column distinguishes asllex (carefully ASL-LEX-aligned),
# freetext (one-word annotations from random signers), and signbank (SignBank
# entries). asllex is the cleanest. We accept all by default but warn when
# fallback types are dominant for a given sign.
DEFAULT_PREFERRED_LABEL_TYPE = "asllex"

# Real Sem-Lex column names (verified from semlex_metadata.csv on 2026-05-23):
# ['', 'video_id', 'signer_id', 'duration', 'split', 'label_type', 'label',
#  'Handshape', 'Selected Fingers', ...].
# We probe broadly so the script also handles future Sem-Lex schema tweaks
# or other ASL datasets with similar shape.
FILENAME_COLS = ("video_id", "filename", "file", "video", "video_file", "clip", "clip_id", "name", "id")
GLOSS_COLS = ("label", "gloss", "sign", "class", "lemma", "entry_id", "EntryID")
SIGNER_COLS = ("signer_id", "signer", "participant", "participant_id", "subject", "subject_id")
SPLIT_COLS = ("split", "set", "partition", "fold")
LABEL_TYPE_COLS = ("label_type",)


def load_wave1_signs() -> list[str]:
    with open(WAVE1_CSV, newline="", encoding="utf-8") as f:
        return [row["sign_id"] for row in csv.DictReader(f)]


def candidate_glosses(sign_id: str, override: dict[str, list[str]]) -> set[str]:
    """Lowercase Sem-Lex labels that should map to this sign_id.

    Sem-Lex labels are lowercase. Always include `{sign_id}` plus Sem-Lex's
    `{sign_id}_<digit>` disambiguation pattern (eat_1, deaf_2, what_1, etc.).
    Override entries supply hand-curated extra lemmas (e.g. goodbye → bye).
    Caller is responsible for matching with a regex that handles the _<digit>
    suffix when this set returns only the bare sign_id.
    """
    if sign_id in override:
        return {g.lower() for g in override[sign_id]}
    return {sign_id.lower()}


# Matches `sign_id` or `sign_id_<one_or_more_digits>` (Sem-Lex variant suffix).
def _matches_any(label: str, candidates: set[str]) -> bool:
    label = label.lower()
    if label in candidates:
        return True
    # Strip _<digits> suffix and re-check (catches eat_1, deaf_2, etc. when the
    # candidates set has just "eat" / "deaf").
    if "_" in label:
        head, _, tail = label.rpartition("_")
        if tail.isdigit() and head in candidates:
            return True
    return False


def pick_column(fieldnames: Iterable[str], candidates: Iterable[str]) -> str | None:
    lower_to_orig = {f.lower(): f for f in fieldnames}
    for c in candidates:
        if c.lower() in lower_to_orig:
            return lower_to_orig[c.lower()]
    return None


def _gdown_to(file_id: str, dest: Path) -> Path:
    """Download a small Google Drive file (used only for the metadata CSV).

    For large tarballs see `open_drive_download_stream()` — those are streamed
    directly into tarfile to avoid landing them on disk (the train tarball
    alone is 23.7 GB, larger than the GitHub-hosted runner's ephemeral disk).
    """
    import gdown
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading {file_id} → {dest}")
    gdown.download(url, str(dest), quiet=False, resume=True)
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"gdown produced empty/missing file at {dest}")
    print(f"  done: {dest.stat().st_size / 1e6:.1f} MB")
    return dest


_FORM_ACTION_RE = re.compile(
    r'<form\b[^>]*\bid="download-form"[^>]*\baction="([^"]+)"', re.IGNORECASE
)
_FORM_BLOCK_RE = re.compile(
    r'<form\b[^>]*\bid="download-form"[^>]*>(.*?)</form>', re.IGNORECASE | re.DOTALL
)
# Captures any <input name="X" value="Y"> regardless of attribute order/quoting.
_INPUT_RE = re.compile(
    r'<input\b[^>]*?\bname="([^"]+)"[^>]*?\bvalue="([^"]*)"', re.IGNORECASE
)
_INPUT_RE_REVERSED = re.compile(
    r'<input\b[^>]*?\bvalue="([^"]*)"[^>]*?\bname="([^"]+)"', re.IGNORECASE
)


def _parse_drive_confirm_form(html: str) -> tuple[str, dict[str, str]] | None:
    """Extract (action_url, form_field_dict) from Drive's current download interstitial.

    The current Drive UX renders a `<form id="download-form" action="...">` whose
    action URL points at drive.usercontent.google.com (not drive.google.com).
    Hidden inputs carry id/export/authuser/confirm/uuid. We just submit whatever
    the form says — don't second-guess the field set.
    """
    action_match = _FORM_ACTION_RE.search(html)
    block_match = _FORM_BLOCK_RE.search(html)
    if not action_match or not block_match:
        return None
    action = action_match.group(1).replace("&amp;", "&")
    fields: dict[str, str] = {}
    for m in _INPUT_RE.finditer(block_match.group(1)):
        fields[m.group(1)] = m.group(2)
    for m in _INPUT_RE_REVERSED.finditer(block_match.group(1)):
        fields.setdefault(m.group(2), m.group(1))
    return action, fields


def open_drive_download_stream(file_id: str):
    """Return a streaming requests.Response for a Google Drive file.

    Handles Drive's "virus scan can't run" interstitial for files > ~100 MB via:
      1. Legacy `download_warning_*` cookie (older UX)
      2. The HTML form's own action URL + hidden inputs (current UX) — critical:
         the action URL is drive.usercontent.google.com, NOT drive.google.com/uc.
    Raises RuntimeError with a body preview if neither path yields binary data.
    """
    import requests
    session = requests.Session()
    base = "https://drive.google.com/uc"
    params = {"export": "download", "id": file_id}
    resp = session.get(base, params=params, stream=True, allow_redirects=True)

    def _looks_like_binary(r) -> bool:
        ctype = r.headers.get("Content-Type", "").lower()
        clen_str = r.headers.get("Content-Length", "")
        try:
            clen = int(clen_str) if clen_str else 0
        except ValueError:
            clen = 0
        # Drive's HTML interstitial is text/html and small (a few KB).
        # Real tarball is application/octet-stream or similar, ≫1 MB.
        return ("text/html" not in ctype) and (clen > 1_000_000 or clen == 0 and "octet" in ctype)

    if not _looks_like_binary(resp):
        # Try cookie-based confirm (legacy)
        token = next(
            (v for k, v in session.cookies.items() if k.startswith("download_warning")),
            None,
        )
        if token:
            params["confirm"] = token
            resp.close()
            resp = session.get(base, params=params, stream=True, allow_redirects=True)

        if not _looks_like_binary(resp):
            # Parse the new-UX form and resubmit to its action URL.
            html = resp.text
            parsed = _parse_drive_confirm_form(html)
            if parsed is None:
                preview = html[:600].replace("\n", " ")
                raise RuntimeError(
                    f"Drive did not return a download form for {file_id}. "
                    f"Content-Type={resp.headers.get('Content-Type')!r} "
                    f"Content-Length={resp.headers.get('Content-Length')!r}. "
                    f"Body preview: {preview!r}"
                )
            action_url, fields = parsed
            print(f"  Drive interstitial: posting to {action_url} with fields={sorted(fields.keys())}")
            resp.close()
            resp = session.get(action_url, params=fields, stream=True, allow_redirects=True)

    resp.raise_for_status()
    # Disable HTTP-level auto-decompression. Drive shouldn't send
    # Content-Encoding: gzip for a .tar.gz file, but if it ever does
    # we want the raw bytes so tarfile's own gunzip works correctly.
    resp.raw.decode_content = False
    return resp


class _ChunkedFile:
    """Wraps `iter_content` chunks into a `read(n)`-style file for tarfile.

    Using iter_content avoids subtle issues with urllib3's HTTPResponse buffering
    and lets us cleanly prepend a peeked chunk back to the stream when we need
    to inspect the magic bytes before handing the stream off.
    """

    def __init__(self, response, chunk_size: int = 1024 * 1024, prefix: bytes = b""):
        self._it = response.iter_content(chunk_size=chunk_size, decode_unicode=False)
        self._buffer = prefix

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            chunks = [self._buffer]
            self._buffer = b""
            for c in self._it:
                if c:
                    chunks.append(c)
            return b"".join(chunks)
        while len(self._buffer) < n:
            try:
                nxt = next(self._it)
            except StopIteration:
                break
            if nxt:
                self._buffer += nxt
        out, self._buffer = self._buffer[:n], self._buffer[n:]
        return out


def load_metadata(csv_path: Path, wave1: list[str], override: dict[str, list[str]]
                  ) -> dict[str, tuple[str, str, str]]:
    """Return {video_id_or_filename: (sign_id, signer_id, split)} for matching rows."""
    wanted: dict[str, set[str]] = {s: candidate_glosses(s, override) for s in wave1}

    matches: dict[str, tuple[str, str, str]] = {}
    per_sign_count: dict[str, int] = {s: 0 for s in wave1}
    label_type_breakdown: dict[str, dict[str, int]] = {s: {} for s in wave1}

    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fname_col = pick_column(reader.fieldnames or [], FILENAME_COLS)
        gloss_col = pick_column(reader.fieldnames or [], GLOSS_COLS)
        signer_col = pick_column(reader.fieldnames or [], SIGNER_COLS)
        split_col = pick_column(reader.fieldnames or [], SPLIT_COLS)
        ltype_col = pick_column(reader.fieldnames or [], LABEL_TYPE_COLS)
        if not fname_col or not gloss_col:
            raise SystemExit(
                f"Could not find filename + gloss columns in {csv_path}.\n"
                f"Available columns: {reader.fieldnames}\n"
                f"Probed filename names: {FILENAME_COLS}\n"
                f"Probed gloss names: {GLOSS_COLS}\n"
                f"Either rename the column in the CSV or add to the *_COLS lists in fetch_semlex.py."
            )
        print(f"Metadata columns → filename={fname_col!r} gloss={gloss_col!r} "
              f"signer={signer_col!r} split={split_col!r} label_type={ltype_col!r}")

        unique_glosses: set[str] = set()
        for row in reader:
            gloss_raw = (row.get(gloss_col) or "").strip()
            if not gloss_raw:
                continue
            unique_glosses.add(gloss_raw)
            matched: str | None = None
            for sign_id, variants in wanted.items():
                if _matches_any(gloss_raw, variants):
                    matched = sign_id
                    break
            if not matched:
                continue
            fname = (row.get(fname_col) or "").strip()
            if not fname:
                continue
            # Sem-Lex's `video_id` is a bare alphanumeric (no extension).
            # The tar member basenames are `<video_id>.mp4`. Store the raw
            # value here; stream_extract probes both basename AND stem.
            key = Path(fname).name
            signer_raw = (row.get(signer_col) or "unknown").strip() if signer_col else "unknown"
            # Must match import_captures.py's regex ^(.+)_(signer_[a-z0-9]+)_\d+$.
            # The signer token has to be `signer_` + alphanumerics ONLY (no extra
            # underscores), so strip-not-replace non-alnum characters.
            signer_clean = re.sub(r"[^A-Za-z0-9]+", "", signer_raw).lower() or "unknown"
            signer_id = f"signer_semlex{signer_clean}"
            split = (row.get(split_col) or "unknown").strip() if split_col else "unknown"
            matches[key] = (matched, signer_id, split)
            per_sign_count[matched] = per_sign_count.get(matched, 0) + 1
            if ltype_col:
                lt = (row.get(ltype_col) or "unknown").strip()
                label_type_breakdown[matched][lt] = label_type_breakdown[matched].get(lt, 0) + 1

    matched_signs = sum(1 for n in per_sign_count.values() if n > 0)
    print(f"\nMetadata: matched {len(matches)} rows covering {matched_signs}/{len(wave1)} signs "
          f"(out of {len(unique_glosses)} unique Sem-Lex glosses)")
    print("Per-sign coverage in metadata (before per-sign cap applied):")
    for s in wave1:
        n = per_sign_count.get(s, 0)
        breakdown = label_type_breakdown.get(s, {})
        breakdown_str = " ".join(f"{k}={v}" for k, v in sorted(breakdown.items())) or "n/a"
        flag = " ⚠ LOW" if n < 10 else ""
        flag += " ⚠ KNOWN-GAP" if s in KNOWN_LOW_COVERAGE else ""
        print(f"  {s:20s} {n:>5d}   [{breakdown_str}]{flag}")
    return matches


def _extract_from_tar_stream(
    tar_stream,
    role_label: str,
    wanted: dict[str, tuple[str, str, str]],
    cap_per_sign: int,
    out_dir: Path,
    existing_counts: dict[str, int] | None = None,
) -> int:
    """Extract matched videos from any streaming tar.gz source.

    Works with both an on-disk file (passed via tarfile.open(path, mode='r|gz'))
    and a live HTTP response (passed via tarfile.open(fileobj=..., mode='r|gz')).
    The mode='r|gz' (with the pipe) is the key — sequential, no seek required.
    """
    counts: dict[str, int] = dict(existing_counts) if existing_counts else {}
    extracted = 0
    sample_members: list[str] = []
    members_scanned = 0
    all_signs_capped_logged = False

    for i, member in enumerate(tar_stream, start=1):
        members_scanned = i
        if i % 5000 == 0:
            print(f"  scanned {i} members ({extracted} extracted so far) — "
                  f"caps hit: {sum(1 for c in counts.values() if c >= cap_per_sign)}/{len(wanted) and len(set(m[0] for m in wanted.values()))}")
        if not member.isfile():
            continue
        if len(sample_members) < 5:
            sample_members.append(member.name)
        mem_path = Path(member.name)
        meta = wanted.get(mem_path.name) or wanted.get(mem_path.stem)
        if not meta:
            continue
        sign_id, signer_id, _split = meta
        if counts.get(sign_id, 0) >= cap_per_sign:
            continue
        ext = mem_path.suffix.lower() or ".mp4"
        counts[sign_id] = counts.get(sign_id, 0) + 1
        target = out_dir / f"{sign_id}_{signer_id}_{counts[sign_id]:04d}{ext}"
        src = tar_stream.extractfile(member)
        if src is None:
            continue
        with src, open(target, "wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
        extracted += 1

        # Early-stop optimization: if every sign in `wanted` has hit its cap,
        # we can stop scanning the rest of the tarball. Especially valuable
        # for the 23 GB train tarball — Sem-Lex's videos appear in a roughly
        # gloss-ordered layout, so most caps fill in the first few GB.
        unique_signs = {m[0] for m in wanted.values()}
        if all(counts.get(s, 0) >= cap_per_sign for s in unique_signs):
            if not all_signs_capped_logged:
                print(f"  ✓ all {len(unique_signs)} signs hit cap_per_sign={cap_per_sign}; "
                      f"stopping {role_label} scan early (saves bandwidth + disk)")
                all_signs_capped_logged = True
            break

    print(f"  done {role_label}: scanned {members_scanned} members, extracted {extracted}")
    if extracted == 0:
        print(f"  ⚠ ZERO extracted from {role_label}. First {len(sample_members)} tar member names:", file=sys.stderr)
        for name in sample_members:
            print(f"    {name!r}  (basename={Path(name).name!r} stem={Path(name).stem!r})", file=sys.stderr)
        print(f"  ⚠ Metadata expects keys like (first 3): {list(wanted.keys())[:3]}", file=sys.stderr)
    if existing_counts is not None:
        existing_counts.update(counts)
    return extracted


def stream_extract_from_drive(
    file_id: str,
    role_label: str,
    wanted: dict[str, tuple[str, str, str]],
    cap_per_sign: int,
    out_dir: Path,
    existing_counts: dict[str, int] | None = None,
) -> int:
    """Streaming download + streaming tar extract — the tarball never lands on disk.

    Peak disk = total bytes of *matched* videos only (~1–2 GB). Source tarball
    can be arbitrarily large (Sem-Lex's train.tar.gz is 23.7 GB).
    """
    print(f"\nStreaming {role_label}.tar.gz (Drive ID {file_id[:8]}…) — never saved to disk")
    resp = open_drive_download_stream(file_id)
    total_bytes = resp.headers.get("Content-Length")
    if total_bytes:
        try:
            n = int(total_bytes)
            print(f"  Source size: {n / 1e9:.2f} GB (streamed, not stored)")
        except ValueError:
            print(f"  Source size: {total_bytes!r} (streamed, not stored)")
    else:
        print("  Source size: (Content-Length not provided; streaming anyway)")
    try:
        # Peek the gzip magic bytes BEFORE opening tarfile so we can fail with
        # a clear error if Drive handed us HTML instead of a tarball.
        first_chunk_iter = resp.iter_content(chunk_size=1024 * 1024, decode_unicode=False)
        first = b""
        for c in first_chunk_iter:
            if c:
                first = c
                break
        if first[:2] != b"\x1f\x8b":
            preview = first[:300]
            try:
                preview_str = preview.decode("utf-8", errors="replace")
            except Exception:
                preview_str = repr(preview)
            raise RuntimeError(
                f"Response for {role_label} is not a gzip stream. "
                f"Content-Type={resp.headers.get('Content-Type')!r} "
                f"first {len(first)} bytes hex={first[:16].hex()!r}. "
                f"Body preview: {preview_str!r}"
            )
        # Reconstruct the stream with the peeked chunk prepended.
        stream = _ChunkedFile(resp, chunk_size=1024 * 1024, prefix=first)
        with tarfile.open(fileobj=stream, mode="r|gz") as tar:
            return _extract_from_tar_stream(
                tar, role_label, wanted, cap_per_sign, out_dir, existing_counts
            )
    finally:
        resp.close()


def stream_extract(tar_path: Path, wanted: dict[str, tuple[str, str, str]],
                   cap_per_sign: int, out_dir: Path,
                   existing_counts: dict[str, int] | None = None) -> int:
    """Stream-extract from an on-disk tar.gz (used for local testing only).

    The production CI path uses `stream_extract_from_drive()` which never
    writes the tarball to disk. This wrapper exists so a previously-downloaded
    local tarball can still be re-processed without going over the wire.
    """
    print(f"\nStream-extracting on-disk {tar_path.name}...")
    with tarfile.open(tar_path, mode="r|gz") as tar:
        return _extract_from_tar_stream(
            tar, tar_path.name, wanted, cap_per_sign, out_dir, existing_counts
        )


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
    global_counts: dict[str, int] = {}
    for role in ("train", "val", "test"):
        if role not in drive_files:
            print(f"Skipping {role} (not in SEMLEX_DRIVE_FILES).")
            continue
        n = stream_extract_from_drive(
            drive_files[role], role, wanted, args.clips_per_sign,
            out_dir, existing_counts=global_counts,
        )
        total_extracted += n

    print(f"\nTotal extracted: {total_extracted} videos → {out_dir}")
    print("Per-sign extracted count (after per-sign cap):")
    for sign_id in wave1:
        n = global_counts.get(sign_id, 0)
        flag = " ⚠ ZERO" if n == 0 else (" ⚠ LOW" if n < 5 else "")
        flag += " (KNOWN-GAP)" if sign_id in KNOWN_LOW_COVERAGE else ""
        print(f"  {sign_id:20s} {n:>5d}{flag}")

    zero_coverage = [s for s in wave1 if global_counts.get(s, 0) == 0]
    if zero_coverage:
        print(f"\n⚠ {len(zero_coverage)} sign(s) extracted ZERO Sem-Lex videos: {zero_coverage}")
        print("  → If unexpected, check the metadata CSV's gloss column and extend GLOSS_MAP_DEFAULT.")
        print("  → Combine with --dataset_source=both at workflow level to backfill from learner_samples.")

    # Fail loudly if we extracted nothing at all — keeps the workflow from
    # silently progressing through decode + manifest steps that would then
    # break far downstream.
    if total_extracted == 0:
        print("\nFATAL: extracted 0 videos total. Likely causes:", file=sys.stderr)
        print("  - Tarball member basenames do not match metadata video_ids", file=sys.stderr)
        print("  - All matched videos were in splits not requested by SEMLEX_DRIVE_FILES", file=sys.stderr)
        print("  - SEMLEX_CLIPS_PER_SIGN cap is 0", file=sys.stderr)
        return 3

    print("\nNext: python ml/scripts/import_captures.py && python ml/scripts/build_manifest.py --wave1 --signer-disjoint")
    return 0


if __name__ == "__main__":
    sys.exit(main())
