"""Report split/sign/signer coverage for an ASL manifest.

This is a pre-training diagnostic. It does not read video frames; it answers:
- Do all trained signs have train/val/test coverage?
- Which signs have zero held-out test support?
- Which signers are assigned to each split?
- Is the nominal signer-disjoint split actually signer-disjoint?
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _pct(n: int, d: int) -> str:
    return "0.0%" if d == 0 else f"{(100 * n / d):.1f}%"


def build_report(manifest: dict) -> dict:
    clips = manifest.get("clips", [])
    sign_ids = sorted(set(manifest.get("sign_ids", [])) or {row["sign_id"] for row in clips})
    splits = ["train", "val", "test"]

    split_counts = Counter(row["split"] for row in clips)
    sign_counts = {split: Counter() for split in splits}
    signer_counts = {split: Counter() for split in splits}
    signers_by_split = {split: set() for split in splits}
    splits_by_signer: dict[str, set[str]] = defaultdict(set)

    for row in clips:
        split = row["split"]
        sign_id = row["sign_id"]
        signer_id = row["signer_id"]
        if split in sign_counts:
            sign_counts[split][sign_id] += 1
            signer_counts[split][signer_id] += 1
            signers_by_split[split].add(signer_id)
        splits_by_signer[signer_id].add(split)

    zero_support = {
        split: [sign_id for sign_id in sign_ids if sign_counts[split][sign_id] == 0]
        for split in splits
    }
    signer_overlap = {
        signer: sorted(splits_seen)
        for signer, splits_seen in sorted(splits_by_signer.items())
        if len(splits_seen) > 1
    }
    test_signers = sorted(signers_by_split["test"])
    train_val_signers = sorted(signers_by_split["train"] | signers_by_split["val"])
    disjoint_ok = not (set(test_signers) & set(train_val_signers))

    return {
        "total_clips": len(clips),
        "num_signs": len(sign_ids),
        "split_counts": dict(split_counts),
        "split_percentages": {
            split: _pct(split_counts.get(split, 0), len(clips)) for split in splits
        },
        "zero_support": zero_support,
        "num_zero_support": {split: len(zero_support[split]) for split in splits},
        "test_signers": test_signers,
        "train_val_signers": train_val_signers,
        "num_signers_by_split": {
            split: len(signers_by_split[split]) for split in splits
        },
        "signer_disjoint_ok": disjoint_ok,
        "signer_overlap": signer_overlap,
        "per_sign": {
            sign_id: {split: sign_counts[split][sign_id] for split in splits}
            for sign_id in sign_ids
        },
    }


def print_report(report: dict, *, max_rows: int) -> None:
    print("Manifest coverage report")
    print(f"Total clips: {report['total_clips']}")
    print(f"Signs: {report['num_signs']}")
    print(f"Splits: {report['split_counts']} ({report['split_percentages']})")
    print(f"Signers by split: {report['num_signers_by_split']}")
    print(f"Signer-disjoint OK: {report['signer_disjoint_ok']}")
    if report["signer_overlap"]:
        print(f"WARNING: signer(s) appear in multiple splits: {report['signer_overlap']}")
    print(f"Test signer IDs: {report['test_signers']}")

    for split in ["train", "val", "test"]:
        missing = report["zero_support"][split]
        print(f"Zero-{split}-support signs: {len(missing)}")
        if missing:
            print("  " + ", ".join(missing))

    rows = sorted(
        report["per_sign"].items(),
        key=lambda item: (item[1]["test"], item[1]["val"], item[1]["train"], item[0]),
    )
    print("\nLowest held-out support signs:")
    print("sign_id train val test")
    for sign_id, counts in rows[:max_rows]:
        print(f"{sign_id:20s} {counts['train']:5d} {counts['val']:3d} {counts['test']:4d}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="ml/data/manifest.json")
    parser.add_argument("--out-json", help="Optional path to write machine-readable report.")
    parser.add_argument("--max-rows", type=int, default=15)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    report = build_report(manifest)
    print_report(report, max_rows=args.max_rows)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
