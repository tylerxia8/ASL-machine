"""Quality checks for manifest clips before landmark training.

The report is intentionally conservative. It flags files that are missing,
malformed, duplicated by source path, unusually still, or low-coverage by sign.
It works with both frame-based NPZ clips and landmark-style NPZ clips.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _numeric_arrays(npz: Any) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for key in npz.files:
        value = npz[key]
        if np.issubdtype(value.dtype, np.number):
            arrays.append(np.asarray(value, dtype=np.float32))
    return arrays


def _motion_score(arr: np.ndarray) -> float | None:
    if arr.ndim < 2:
        return None
    frame_axis = 0
    if arr.shape[0] < 2:
        return 0.0
    diffs = np.diff(arr, axis=frame_axis)
    return float(np.nanmean(np.abs(diffs)))


def _inspect_clip(path: Path) -> dict:
    if not path.exists():
        return {"ok": False, "reason": "missing_file"}
    try:
        with np.load(path, allow_pickle=False) as npz:
            arrays = _numeric_arrays(npz)
            if not arrays:
                return {"ok": False, "reason": "no_numeric_arrays"}
            primary = max(arrays, key=lambda arr: arr.size)
            if primary.size == 0:
                return {"ok": False, "reason": "empty_array"}
            if not np.isfinite(primary).all():
                return {"ok": False, "reason": "nan_or_inf"}
            return {
                "ok": True,
                "shape": list(primary.shape),
                "motion": _motion_score(primary),
                "mean": float(np.mean(primary)),
                "std": float(np.std(primary)),
            }
    except Exception as exc:  # noqa: BLE001 - report file-level failures.
        return {"ok": False, "reason": "load_error", "error": str(exc)}


def build_report(manifest: dict, *, root: Path, min_per_sign: int, low_motion: float) -> dict:
    clips = manifest.get("clips", [])
    sign_counts: Counter[str] = Counter()
    signer_counts: Counter[str] = Counter()
    path_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    issues: list[dict] = []
    motion_by_sign: dict[str, list[float]] = defaultdict(list)

    for row in clips:
        sign_id = row.get("sign_id", "unknown")
        signer_id = row.get("signer_id", "unknown")
        rel_path = row.get("path", "")
        path = Path(rel_path)
        if not path.is_absolute():
            path = root / path
        sign_counts[sign_id] += 1
        signer_counts[signer_id] += 1
        path_counts[str(path)] += 1

        inspection = _inspect_clip(path)
        if not inspection["ok"]:
            issues.append({"path": rel_path, "sign_id": sign_id, **inspection})
            continue
        motion = inspection.get("motion")
        if isinstance(motion, float):
            motion_by_sign[sign_id].append(motion)
            if motion <= low_motion:
                issues.append({
                    "path": rel_path,
                    "sign_id": sign_id,
                    "reason": "low_motion",
                    "motion": motion,
                    "shape": inspection.get("shape"),
                })

        try:
            with np.load(path, allow_pickle=False) as npz:
                if "source_path" in npz.files:
                    source_counts[str(npz["source_path"])] += 1
        except Exception:
            pass

    duplicates = [
        {"path": path, "count": count}
        for path, count in path_counts.items()
        if count > 1
    ]
    duplicate_sources = [
        {"source_path": source, "count": count}
        for source, count in source_counts.items()
        if source and count > 1
    ]
    undercovered = [
        {"sign_id": sign_id, "clips": count}
        for sign_id, count in sorted(sign_counts.items())
        if count < min_per_sign
    ]

    return {
        "total_clips": len(clips),
        "num_signs": len(sign_counts),
        "num_signers": len(signer_counts),
        "issues": issues,
        "num_issues": len(issues),
        "duplicates": duplicates,
        "duplicate_sources": duplicate_sources,
        "undercovered_signs": undercovered,
        "motion_by_sign": {
            sign_id: {
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for sign_id, values in sorted(motion_by_sign.items())
            if values
        },
    }


def print_report(report: dict, *, max_rows: int) -> None:
    print("Clip QA report")
    print(f"Clips: {report['total_clips']} | signs: {report['num_signs']} | signers: {report['num_signers']}")
    print(f"Issues: {report['num_issues']} | duplicate paths: {len(report['duplicates'])} | duplicate sources: {len(report['duplicate_sources'])}")
    if report["undercovered_signs"]:
        print("Undercovered signs:")
        for row in report["undercovered_signs"][:max_rows]:
            print(f"  {row['sign_id']}: {row['clips']} clips")
    if report["issues"]:
        print("Top issues:")
        for row in report["issues"][:max_rows]:
            detail = f" motion={row['motion']:.6f}" if isinstance(row.get("motion"), float) else ""
            print(f"  {row['reason']}: {row['path']}{detail}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="ml/data/manifest.json")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out-json", default="ml/exports/clip_qa_report.json")
    parser.add_argument("--min-per-sign", type=int, default=20)
    parser.add_argument("--low-motion", type=float, default=0.0005)
    parser.add_argument("--max-rows", type=int, default=20)
    args = parser.parse_args()

    root = Path(args.repo_root)
    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    report = build_report(manifest, root=root, min_per_sign=args.min_per_sign, low_motion=args.low_motion)
    print_report(report, max_rows=args.max_rows)

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
