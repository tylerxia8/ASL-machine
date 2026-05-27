"""Build manifest.json from clips; optional Wave 1 filter and signer-disjoint test split."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
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


def choose_test_signers(clips: list[dict], target_count: int) -> set[str]:
    """Pick signer-disjoint test signers while maximizing sign coverage.

    A sorted-first signer choice is deterministic but brittle: on Sem-Lex it can
    hold out signers who collectively cover only a narrow slice of the Wave 1
    vocabulary. This greedy selector keeps the same signer-disjoint contract
    while preferring signers that add held-out support for signs not yet covered.
    """
    unique_signers = sorted({c["signer_id"] for c in clips})
    if target_count <= 0 or not unique_signers:
        return set()

    by_signer: dict[str, Counter[str]] = defaultdict(Counter)
    for clip in clips:
        by_signer[clip["signer_id"]][clip["sign_id"]] += 1

    selected: set[str] = set()
    covered_signs: set[str] = set()
    if "signer_b" in unique_signers:
        selected.add("signer_b")
        covered_signs.update(by_signer["signer_b"])

    while len(selected) < target_count:
        candidates = [s for s in unique_signers if s not in selected and s != "signer_a"]
        if not candidates:
            break

        def score(signer: str) -> tuple[int, int, int, str]:
            counts = by_signer[signer]
            new_signs = set(counts) - covered_signs
            new_clip_count = sum(counts[sign] for sign in new_signs)
            total_clip_count = sum(counts.values())
            return (len(new_signs), new_clip_count, total_clip_count, signer)

        best = max(candidates, key=score)
        selected.add(best)
        covered_signs.update(by_signer[best])
    return selected


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
        print("No clips found. Import captures first.", file=__import__("sys").stderr)
        raise SystemExit(2)

    manifest_clips = []
    if args.signer_disjoint:
        # Hold out a deterministic ~15% of unique signers as the test set
        # (signer-disjoint by construction — the rubric Req 8 metric).
        # Always include `signer_b` as test if present (legacy 2-signer convention).
        unique_signers = sorted({c["signer_id"] for c in clips})
        n_test = max(1, int(round(len(unique_signers) * 0.15))) if len(unique_signers) > 1 else 0
        test_signers = choose_test_signers(clips, n_test)

        if not test_signers:
            # Only 1 unique signer (or the only available holdouts are excluded).
            # Signer-disjoint isn't possible — fall back to a within-signer
            # per-clip random split so eval still has data. The validation
            # report should flag that the resulting test accuracy is NOT
            # signer-disjoint (within-signer is a weaker metric).
            print(f"WARNING: --signer-disjoint with {len(unique_signers)} unique signer(s) → "
                  f"falling back to per-clip random {1 - args.test_ratio - args.val_ratio:.0%}/"
                  f"{args.val_ratio:.0%}/{args.test_ratio:.0%} train/val/test split. "
                  f"This is NOT signer-disjoint. Add more signers to get a Req-8-style eval.",
                  file=__import__("sys").stderr)
            by_sign = {}
            for c in clips:
                by_sign.setdefault(c["sign_id"], []).append(c)
            for sign_id, group in sorted(by_sign.items()):
                random.shuffle(group)
                n = len(group)
                n_te = max(1, int(n * args.test_ratio)) if n >= 3 else (1 if n >= 2 else 0)
                n_va = max(1, int(n * args.val_ratio)) if n >= 3 else 0
                for i, c in enumerate(group):
                    if i < n_te:
                        split = "test"
                    elif i < n_te + n_va:
                        split = "val"
                    else:
                        split = "train"
                    manifest_clips.append({**c, "split": split})
        else:
            print(f"Signer-disjoint split: {len(test_signers)} test signer(s) "
                  f"out of {len(unique_signers)} unique: {sorted(test_signers)}")
            for c in clips:
                if c["signer_id"] in test_signers:
                    split = "test"
                else:
                    split = "train" if random.random() < 0.85 else "val"
                manifest_clips.append({**c, "split": split})
            test_signs = {c["sign_id"] for c in manifest_clips if c["split"] == "test"}
            missing_test = sorted(sign_ids - test_signs)
            if missing_test:
                print(f"WARNING: {len(missing_test)} sign(s) have zero signer-disjoint "
                      f"test support: {missing_test}")
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
