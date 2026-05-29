"""Compare two eval_metrics.json files and recommend whether to promote.

This is intentionally conservative: a candidate can improve overall accuracy
but still be flagged if key signs regress too far.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

KEY_SIGNS = {"five", "four", "how", "who", "please", "two", "deaf"}


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _f1(metrics: dict, sign_id: str) -> float:
    return float(metrics.get("report", {}).get(sign_id, {}).get("f1-score", 0.0))


def _support(metrics: dict, sign_id: str) -> int:
    return int(metrics.get("report", {}).get(sign_id, {}).get("support", 0))


def compare(base: dict, candidate: dict, max_key_drop: float, max_any_drop: float) -> tuple[bool, str]:
    base_report = base.get("report", {})
    cand_report = candidate.get("report", {})
    sign_ids = sorted(set(base.get("sign_ids", [])) | set(candidate.get("sign_ids", [])))
    rows = []
    blockers = []

    for sign_id in sign_ids:
      if sign_id not in base_report or sign_id not in cand_report:
          continue
      before = _f1(base, sign_id)
      after = _f1(candidate, sign_id)
      delta = after - before
      support = _support(candidate, sign_id)
      rows.append((delta, sign_id, before, after, support))
      if sign_id in KEY_SIGNS and delta < -max_key_drop:
          blockers.append(f"`{sign_id}` key-sign F1 dropped {abs(delta):.3f}")
      elif delta < -max_any_drop and support >= 10:
          blockers.append(f"`{sign_id}` F1 dropped {abs(delta):.3f}")

    base_acc = float(base.get("accuracy", 0.0))
    cand_acc = float(candidate.get("accuracy", 0.0))
    base_macro = float(base_report.get("macro avg", {}).get("f1-score", 0.0))
    cand_macro = float(cand_report.get("macro avg", {}).get("f1-score", 0.0))
    base_weighted = float(base_report.get("weighted avg", {}).get("f1-score", 0.0))
    cand_weighted = float(cand_report.get("weighted avg", {}).get("f1-score", 0.0))

    promote = cand_acc >= base_acc and cand_weighted >= base_weighted and not blockers
    lines = [
        f"# Model Comparison: `{base.get('model_version', 'base')}` vs `{candidate.get('model_version', 'candidate')}`",
        "",
        "| Metric | Base | Candidate | Delta |",
        "|---|---:|---:|---:|",
        f"| Accuracy | {base_acc:.4f} | {cand_acc:.4f} | {cand_acc - base_acc:+.4f} |",
        f"| Macro F1 | {base_macro:.4f} | {cand_macro:.4f} | {cand_macro - base_macro:+.4f} |",
        f"| Weighted F1 | {base_weighted:.4f} | {cand_weighted:.4f} | {cand_weighted - base_weighted:+.4f} |",
        "",
        f"**Recommendation:** {'PROMOTE' if promote else 'DO NOT PROMOTE'}",
        "",
    ]
    if blockers:
        lines.append("## Blockers")
        lines.append("")
        lines.extend(f"- {item}" for item in blockers)
        lines.append("")

    lines.extend([
        "## Largest Per-Sign Changes",
        "",
        "| Sign | Base F1 | Candidate F1 | Delta | Candidate support |",
        "|---|---:|---:|---:|---:|",
    ])
    for delta, sign_id, before, after, support in sorted(rows, key=lambda row: row[0])[:8]:
        lines.append(f"| `{sign_id}` | {before:.3f} | {after:.3f} | {delta:+.3f} | {support} |")
    for delta, sign_id, before, after, support in sorted(rows, key=lambda row: row[0], reverse=True)[:8]:
        lines.append(f"| `{sign_id}` | {before:.3f} | {after:.3f} | {delta:+.3f} | {support} |")
    lines.append("")
    return promote, "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out")
    parser.add_argument("--max-key-drop", type=float, default=0.12)
    parser.add_argument("--max-any-drop", type=float, default=0.20)
    parser.add_argument("--fail-on-no-promote", action="store_true")
    args = parser.parse_args()

    promote, report = compare(_load(args.base), _load(args.candidate), args.max_key_drop, args.max_any_drop)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report + "\n", encoding="utf-8")
        print(f"Wrote {out}")
    else:
        print(report)
    return 1 if args.fail_on_no_promote and not promote else 0


if __name__ == "__main__":
    raise SystemExit(main())
