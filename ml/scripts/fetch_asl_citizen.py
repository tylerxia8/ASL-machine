"""Fetch a Wave 1 subset from the ASL Citizen zip without downloading 45 GB.

ASL Citizen official download:
https://www.microsoft.com/en-us/research/project/asl-citizen/

The public archive is a large ZIP. This script reads the ZIP64 central
directory using HTTP byte ranges, extracts the small split CSV files, and then
downloads only videos whose glosses match the requested Wave 1 signs.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import struct
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
WAVE1_CSV = ROOT / "content" / "wave1_signs.csv"
INCOMING_DIR = ROOT / "ml" / "data" / "incoming"

ASL_CITIZEN_URL = (
    "https://download.microsoft.com/download/b/8/8/"
    "b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
)

SPLIT_FILES = {
    "train": "ASL_Citizen/splits/train.csv",
    "val": "ASL_Citizen/splits/val.csv",
    "test": "ASL_Citizen/splits/test.csv",
}

GLOSS_MAP_DEFAULT: dict[str, list[str]] = {
    "goodbye": ["GOODBYE", "BYE"],
    "thank_you": ["THANKYOU", "THANKS"],
    "how": ["HOW", "HOW1", "HOW2"],
    "deaf": ["DEAF", "DEAF1", "DEAF2"],
}


@dataclass(frozen=True)
class ZipEntry:
    name: str
    method: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int


def _norm_gloss(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def _request_range(url: str, start: int, end: int, timeout: int = 120) -> bytes:
    res = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=timeout)
    res.raise_for_status()
    if res.status_code != 206:
        raise RuntimeError(f"Server did not honor byte range {start}-{end}; status={res.status_code}")
    return res.content


def _zip64_central_directory(url: str) -> bytes:
    size = int(requests.head(url, timeout=30).headers["Content-Length"])
    tail_size = min(size, 4 * 1024 * 1024)
    tail = _request_range(url, size - tail_size, size - 1)
    eocd_pos = tail.rfind(b"PK\x05\x06")
    locator_pos = tail.rfind(b"PK\x06\x07", 0, eocd_pos)
    if eocd_pos < 0 or locator_pos < 0:
        raise RuntimeError("Could not find ZIP64 end-of-central-directory records")

    _, _, eocd64_offset, _ = struct.unpack("<4sLQL", tail[locator_pos : locator_pos + 20])
    eocd64 = _request_range(url, eocd64_offset, eocd64_offset + 80)
    if eocd64[:4] != b"PK\x06\x06":
        raise RuntimeError("ZIP64 EOCD pointer was invalid")
    body_size = struct.unpack("<Q", eocd64[4:12])[0]
    body = eocd64[12 : 12 + body_size]
    _, _, _, _, _, _, cd_size, cd_offset = struct.unpack("<2H2L4Q", body[:44])
    return _request_range(url, cd_offset, cd_offset + cd_size - 1, timeout=180)


def load_zip_index(url: str = ASL_CITIZEN_URL) -> dict[str, ZipEntry]:
    cd = _zip64_central_directory(url)
    entries: dict[str, ZipEntry] = {}
    pos = 0
    while pos + 46 <= len(cd) and cd[pos : pos + 4] == b"PK\x01\x02":
        fields = struct.unpack("<4s6H3L5H2L", cd[pos : pos + 46])
        method = fields[4]
        compressed_size = fields[8]
        uncompressed_size = fields[9]
        name_len = fields[10]
        extra_len = fields[11]
        comment_len = fields[12]
        local_header_offset = fields[16]
        name = cd[pos + 46 : pos + 46 + name_len].decode("utf-8", errors="replace")
        extra = cd[pos + 46 + name_len : pos + 46 + name_len + extra_len]

        if (
            compressed_size == 0xFFFFFFFF
            or uncompressed_size == 0xFFFFFFFF
            or local_header_offset == 0xFFFFFFFF
        ):
            extra_pos = 0
            while extra_pos + 4 <= len(extra):
                header_id, data_size = struct.unpack("<HH", extra[extra_pos : extra_pos + 4])
                extra_pos += 4
                data = extra[extra_pos : extra_pos + data_size]
                extra_pos += data_size
                if header_id != 0x0001:
                    continue
                data_pos = 0
                if uncompressed_size == 0xFFFFFFFF:
                    uncompressed_size = struct.unpack("<Q", data[data_pos : data_pos + 8])[0]
                    data_pos += 8
                if compressed_size == 0xFFFFFFFF:
                    compressed_size = struct.unpack("<Q", data[data_pos : data_pos + 8])[0]
                    data_pos += 8
                if local_header_offset == 0xFFFFFFFF:
                    local_header_offset = struct.unpack("<Q", data[data_pos : data_pos + 8])[0]

        entries[name] = ZipEntry(name, method, compressed_size, uncompressed_size, local_header_offset)
        pos += 46 + name_len + extra_len + comment_len
    return entries


def extract_entry(url: str, entry: ZipEntry) -> bytes:
    header = _request_range(url, entry.local_header_offset, entry.local_header_offset + 4095)
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"Bad local header for {entry.name}")
    fields = struct.unpack("<4s5H3L2H", header[:30])
    name_len = fields[9]
    extra_len = fields[10]
    data_start = entry.local_header_offset + 30 + name_len + extra_len
    compressed = _request_range(
        url,
        data_start,
        data_start + entry.compressed_size - 1,
        timeout=180,
    )
    if entry.method == 0:
        return compressed
    if entry.method == 8:
        return zlib.decompress(compressed, -15)
    raise RuntimeError(f"Unsupported ZIP compression method {entry.method} for {entry.name}")


def load_wave1_signs() -> list[str]:
    with open(WAVE1_CSV, newline="", encoding="utf-8") as f:
        return [row["sign_id"] for row in csv.DictReader(f)]


def candidate_glosses(sign_id: str, overrides: dict[str, list[str]]) -> set[str]:
    if sign_id in overrides:
        return {_norm_gloss(v) for v in overrides[sign_id]}
    return {_norm_gloss(sign_id)}


def load_split_rows(url: str, entries: dict[str, ZipEntry], splits: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split in splits:
        entry_name = SPLIT_FILES[split]
        text = extract_entry(url, entries[entry_name]).decode("utf-8")
        for row in csv.DictReader(io.StringIO(text)):
            row["split"] = split
            rows.append(row)
    return rows


def select_rows(
    rows: list[dict[str, str]],
    signs: list[str],
    overrides: dict[str, list[str]],
    clips_per_sign: int,
) -> list[tuple[str, dict[str, str]]]:
    wanted = {sign_id: candidate_glosses(sign_id, overrides) for sign_id in signs}
    counts: Counter[str] = Counter()
    selected: list[tuple[str, dict[str, str]]] = []
    coverage: Counter[str] = Counter()

    for row in rows:
        gloss = _norm_gloss(row["Gloss"])
        for sign_id, candidates in wanted.items():
            if gloss in candidates:
                coverage[sign_id] += 1
                if counts[sign_id] < clips_per_sign:
                    selected.append((sign_id, row))
                    counts[sign_id] += 1
                break

    print("ASL Citizen matched rows before cap:")
    for sign_id in signs:
        n = coverage[sign_id]
        flag = " ZERO" if n == 0 else (" LOW" if n < 10 else "")
        print(f"  {sign_id:20s} {n:>4d}{flag}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=ASL_CITIZEN_URL)
    parser.add_argument("--out-dir", default=str(INCOMING_DIR))
    parser.add_argument("--clips-per-sign", type=int, default=30)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--signs", help="Comma-separated sign ids. Default: Wave 1 roster.")
    parser.add_argument("--gloss-map", help="Optional JSON file overriding sign_id -> ASL Citizen gloss list.")
    args = parser.parse_args()

    signs = [s.strip() for s in args.signs.split(",")] if args.signs else load_wave1_signs()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    overrides = dict(GLOSS_MAP_DEFAULT)
    if args.gloss_map:
        overrides.update(json.loads(Path(args.gloss_map).read_text(encoding="utf-8")))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading ASL Citizen ZIP directory via HTTP ranges...")
    entries = load_zip_index(args.url)
    rows = load_split_rows(args.url, entries, splits)
    selected = select_rows(rows, signs, overrides, args.clips_per_sign)

    print(f"\nDownloading {len(selected)} selected ASL Citizen videos -> {out_dir}")
    written = 0
    counts: Counter[str] = Counter()
    for sign_id, row in selected:
        video_name = row["Video file"]
        entry_name = f"ASL_Citizen/videos/{video_name}"
        entry = entries.get(entry_name)
        if entry is None:
            print(f"  missing {entry_name}")
            continue
        participant = re.sub(r"[^a-z0-9]+", "", row["Participant ID"].lower())
        counts[sign_id] += 1
        ext = Path(video_name).suffix.lower() or ".mp4"
        out_path = out_dir / f"{sign_id}_signer_aslcitizen{participant}_{counts[sign_id]:04d}{ext}"
        if not out_path.exists():
            out_path.write_bytes(extract_entry(args.url, entry))
        written += 1
        print(f"  {sign_id:12s} {video_name} -> {out_path.name}")

    print(f"\nDone: {written} files ready for import_captures.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
