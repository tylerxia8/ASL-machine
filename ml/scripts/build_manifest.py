"""Build manifest.json from clips; optional Wave 1 filter and signer-disjoint test split."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLIPS_DIR = ROOT / "ml" / "data" / "clips"
WAVE1 = ROOT / "content" / "wave1_signs.csv"


def load_wave1_ids() -> set[str]:
    ids = set()
    with open(WAVE1, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids.add(row["sign_id"])
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wave1", action="store_true", help="Only include wave1_signs.csv")
    parser.add_argument("--signer-disjoint", action="store_true", help="Hold out signer_b for test")
    args = parser.parse_args()
    random.seed(args.seed)

    allowed = load_wave1_ids() if args.wave1 else None
    clips = []
    sign_ids = set()

    for npz in CLIPS_DIR.rglob("*.npz"):
        parts = npz.relative_to(CLIPS_DIR).parts
        if len(parts) < 3:
            continue
        sign_id, signer_id = parts[0], parts[1]
        if allowed and sign_id not in allowed:
            continue
        sign_ids.add(sign_id)
        clips.append(
            {
                "path": str(npz.relative_to(ROOT)).replace("\\", "/"),
                "sign_id": sign_id,
                "signer_id": signer_id,
            }
        )

    if not clips:
        print("No clips found. Import captures first.")
        return

    manifest_clips = []
    if args.signer_disjoint:
        for sign_id in sorted(sign_ids):
            group = [c for c in clips if c["sign_id"] == sign_id]
            for c in group:
                if c["signer_id"] == "signer_b":
                    split = "test"
                elif c["signer_id"] == "signer_a":
                    split = "train" if random.random() < 0.85 else "val"
                else:
                    split = "val" if random.random() < 0.2 else "train"
                manifest_clips.append({**c, "split": split})
    else:
        by_sign = {}
        for c in clips:
            by_sign.setdefault(c["sign_id"], []).append(c)
        for sign_id, group in sorted(by_sign.items()):
            random.shuffle(group)
            n = len(group)
            n_test = max(1, int(n * args.test_ratio))
            n_val = max(1, int(n * args.val_ratio))
            for i, c in enumerate(group):
                if i < n_test:
                    split = "test"
                elif i < n_test + n_val:
                    split = "val"
                else:
                    split = "train"
                manifest_clips.append({**c, "split": split})

    out = {
        "clips": manifest_clips,
        "sign_ids": sorted(sign_ids),
        "wave1_only": args.wave1,
        "signer_disjoint": args.signer_disjoint,
    }
    out_path = ROOT / "ml" / "data" / "manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Manifest: {len(manifest_clips)} clips, {len(sign_ids)} signs -> {out_path}")


if __name__ == "__main__":
    main()
