"""Build a ranked recording plan from model metrics and existing learner clips.

The recognition model is now mostly data-limited. This script converts
eval_metrics.json into concrete capture targets so the next recording session
focuses on signs that are weak, under-supported, or frequently confused.

Usage:
    python ml/scripts/build_capture_plan.py
    python ml/scripts/build_capture_plan.py --out docs/CAPTURE_PLAN.md
    python ml/scripts/build_capture_plan.py --format csv --out ml/exports/capture_plan.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = ROOT / "ml" / "exports" / "eval_metrics.json"
DEFAULT_OUT = ROOT / "docs" / "CAPTURE_PLAN.md"
LEARNER_ROOT = ROOT / "ml" / "data" / "learner_samples"
INCOMING = ROOT / "ml" / "data" / "incoming"


def _existing_counts() -> Counter[str]:
    counts: Counter[str] = Counter()
    roots = [LEARNER_ROOT, INCOMING]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() not in {".webm", ".mp4", ".mov", ".mkv", ".avi"}:
                continue
            sign_id = path.name.split("_signer_", 1)[0]
            if sign_id:
                counts[sign_id] += 1
    return counts


def _top_confusions(metrics: dict, sign_id: str, limit: int = 3) -> list[tuple[str, int]]:
    sign_ids = metrics.get("sign_ids", [])
    cm = metrics.get("confusion_matrix", [])
    if sign_id not in sign_ids:
        return []
    row_index = sign_ids.index(sign_id)
    row = cm[row_index] if row_index < len(cm) else []
    pairs = []
    for i, count in enumerate(row):
        if i != row_index and count > 0 and i < len(sign_ids):
            pairs.append((sign_ids[i], int(count)))
    pairs.sort(key=lambda item: item[1], reverse=True)
    return pairs[:limit]


def _target_for(f1: float, support: int, existing: int) -> tuple[int, str]:
    if support < 3:
        return 30, "near-zero test support"
    if f1 < 0.40:
        return 30, "very weak F1"
    if f1 < 0.60:
        return 24, "weak F1"
    if f1 < 0.75:
        return 18, "medium F1"
    if existing < 10:
        return 10, "learner baseline"
    return existing, "covered"


def build_rows(metrics: dict) -> list[dict]:
    report = metrics.get("report", {})
    existing_counts = _existing_counts()
    rows = []
    for sign_id in metrics.get("sign_ids", []):
        row = report.get(sign_id, {})
        f1 = float(row.get("f1-score", 0.0))
        recall = float(row.get("recall", 0.0))
        precision = float(row.get("precision", 0.0))
        support = int(row.get("support", 0))
        existing = int(existing_counts.get(sign_id, 0))
        target, reason = _target_for(f1, support, existing)
        needed = max(0, target - existing)
        confusions = _top_confusions(metrics, sign_id)
        rows.append(
            {
                "sign_id": sign_id,
                "f1": f1,
                "recall": recall,
                "precision": precision,
                "support": support,
                "existing_learner_clips": existing,
                "target_learner_clips": target,
                "clips_needed": needed,
                "reason": reason,
                "top_confusions": ", ".join(f"{label} ({count})" for label, count in confusions),
            }
        )
    rows.sort(
        key=lambda row: (
            row["clips_needed"] == 0,
            row["f1"],
            row["support"],
            -row["clips_needed"],
            row["sign_id"],
        )
    )
    return rows


def write_markdown(rows: list[dict], metrics: dict, out_path: Path) -> None:
    total_needed = sum(row["clips_needed"] for row in rows)
    model_version = metrics.get("model_version", "unknown")
    lines = [
        f"# Capture Plan for `{model_version}`",
        "",
        f"Total additional learner clips recommended: **{total_needed}**",
        "",
        "Prioritize rows from top to bottom. Existing clips are counted from "
        "`ml/data/learner_samples/` and `ml/data/incoming/`.",
        "",
        "| Sign | F1 | Recall | Test clips | Existing learner clips | Target | Need | Reason | Top confusions |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        if row["clips_needed"] == 0 and row["reason"] == "covered":
            continue
        lines.append(
            f"| `{row['sign_id']}` | {row['f1']:.2f} | {row['recall']:.2f} | "
            f"{row['support']} | {row['existing_learner_clips']} | "
            f"{row['target_learner_clips']} | {row['clips_needed']} | "
            f"{row['reason']} | {row['top_confusions']} |"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["sign_id"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    args = parser.parse_args()

    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    rows = build_rows(metrics)
    out_path = Path(args.out)
    if args.format == "csv":
        write_csv(rows, out_path)
    else:
        write_markdown(rows, metrics, out_path)

    total_needed = sum(row["clips_needed"] for row in rows)
    print(f"Wrote {out_path} ({total_needed} additional clips recommended)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
