from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from dataset import SignClipDataset
from model import build_model

ML_ROOT = Path(__file__).resolve().parent
ROOT = ML_ROOT.parent

# eval.py only owns the block between these markers. The surrounding narrative
# (model approach, dataset description, known limitations) is hand-edited and
# preserved on every run.
AUTO_START = "<!-- AUTO-METRICS:START -->"
AUTO_END = "<!-- AUTO-METRICS:END -->"


def _build_auto_block(ckpt: dict, sign_ids: list[str], acc: float, report: dict, cm: np.ndarray) -> str:
    lines: list[str] = []
    lines.append(AUTO_START)
    lines.append("")
    lines.append("> This block is overwritten by `python ml/eval.py`. Edit the narrative")
    lines.append("> sections above/below; do not edit between these markers.")
    lines.append("")
    lines.append(f"**Model version:** `{ckpt.get('model_version', 'unknown')}`  ")
    lines.append(f"**Test accuracy (clip-level, signer-disjoint):** {acc:.2%}  ")
    lines.append(f"**Classes:** {len(sign_ids)}  ")
    lines.append(f"**Checkpoint val accuracy:** {ckpt.get('val_accuracy', 'n/a')}  ")
    lines.append(f"**Confusion matrix shape:** {cm.shape[0]}×{cm.shape[1]}")
    lines.append("")
    lines.append("### Per-class metrics")
    lines.append("")
    lines.append("| Sign | Precision | Recall | F1 | Support |")
    lines.append("|------|-----------|--------|------|---------|")
    for sid in sign_ids:
        r = report.get(sid)
        if r:
            lines.append(
                f"| {sid} | {r['precision']:.2f} | {r['recall']:.2f} | {r['f1-score']:.2f} | {int(r['support'])} |"
            )
    avg = report.get("macro avg", {})
    if avg:
        lines.append(
            f"| **macro avg** | {avg['precision']:.2f} | {avg['recall']:.2f} | {avg['f1-score']:.2f} | {int(avg['support'])} |"
        )
    lines.append("")
    lines.append("### Most-confused pairs (top 10)")
    lines.append("")
    confusions: list[tuple[int, str, str]] = []
    for i, sid_true in enumerate(sign_ids):
        for j, sid_pred in enumerate(sign_ids):
            if i != j and cm[i, j] > 0:
                confusions.append((int(cm[i, j]), sid_true, sid_pred))
    confusions.sort(reverse=True)
    if not confusions:
        lines.append("_None — no off-diagonal entries in the confusion matrix._")
    else:
        lines.append("| True → | Predicted | Count |")
        lines.append("|--------|-----------|-------|")
        for count, t, p in confusions[:10]:
            lines.append(f"| {t} | {p} | {count} |")
    lines.append("")
    lines.append(AUTO_END)
    return "\n".join(lines)


def _splice(existing: str, auto_block: str) -> str:
    if AUTO_START in existing and AUTO_END in existing:
        prefix = existing.split(AUTO_START)[0].rstrip() + "\n\n"
        suffix = "\n\n" + existing.split(AUTO_END)[1].lstrip()
        return prefix + auto_block + suffix
    # No markers — append the auto block at the end.
    return existing.rstrip() + "\n\n" + auto_block + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(ML_ROOT / "checkpoints" / "v1" / "best.pt"))
    parser.add_argument("--manifest", default=str(ML_ROOT / "data" / "manifest.json"))
    parser.add_argument("--report", default=str(ROOT / "docs" / "VALIDATION_REPORT.md"))
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sign_ids = ckpt["sign_ids"]
    label_to_idx = ckpt["label_to_idx"]

    model = build_model(num_classes=len(sign_ids))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_ds = SignClipDataset(Path(args.manifest), "test", label_to_idx)
    loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).argmax(1).numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(y.numpy().tolist())

    labels_idx = list(range(len(sign_ids)))
    report = classification_report(
        y_true, y_pred, labels=labels_idx, target_names=sign_ids, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    acc = report["accuracy"]

    auto_block = _build_auto_block(ckpt, sign_ids, acc, report, cm)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    report_path.write_text(_splice(existing, auto_block), encoding="utf-8")

    metrics_path = ML_ROOT / "exports" / "eval_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": acc,
                "report": report,
                "sign_ids": sign_ids,
                "confusion_matrix": cm.tolist(),
                "model_version": ckpt.get("model_version", "unknown"),
            },
            f,
            indent=2,
        )
    print(f"Wrote {report_path} accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
