from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from eval import _build_auto_block, _splice
from landmark_dataset import HandLandmarkDataset
from landmark_model import build_landmark_model

ML_ROOT = Path(__file__).resolve().parent
ROOT = ML_ROOT.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=str(ML_ROOT / "data" / "manifest.json"))
    parser.add_argument("--feature-dir", default=str(ML_ROOT / "data" / "hand_landmarks"))
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sign_ids = ckpt["sign_ids"]
    label_to_idx = ckpt["label_to_idx"]
    ds = HandLandmarkDataset(Path(args.manifest), "test", label_to_idx, Path(args.feature_dir), augment=False)
    if len(ds) == 0:
        raise SystemExit("No usable hand-landmark features for test evaluation.")
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_landmark_model(num_classes=len(sign_ids)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    predictions: list[dict] = []
    with torch.no_grad():
        offset = 0
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred = probs.argmax(axis=1)
            for i, p in enumerate(pred):
                true_idx = int(y[i].item())
                conf = float(probs[i, p])
                y_true.append(true_idx)
                y_pred.append(int(p))
                predictions.append(
                    {
                        "true_label": sign_ids[true_idx],
                        "predicted_label": sign_ids[int(p)],
                        "confidence": conf,
                        "correct": bool(int(p) == true_idx),
                    }
                )
            offset += len(y)

    labels_idx = list(range(len(sign_ids)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels_idx,
        target_names=sign_ids,
        zero_division=0,
        output_dict=True,
    )
    cm_array = confusion_matrix(y_true, y_pred, labels=labels_idx)
    cm = cm_array.tolist()
    accuracy = sum(int(a == b) for a, b in zip(y_true, y_pred)) / max(len(y_true), 1)

    bins = []
    for lo in np.arange(0.0, 1.0, 0.1):
        hi = min(1.0, lo + 0.1)
        selected = [
            p for p in predictions
            if p["confidence"] >= lo and (p["confidence"] < hi or hi == 1.0)
        ]
        correct = sum(1 for p in selected if p["correct"])
        bins.append(
            {
                "range": f"{lo:.1f}-{hi:.1f}",
                "count": len(selected),
                "correct": correct,
                "accuracy": correct / len(selected) if selected else None,
            }
        )

    metrics = {
        "model_version": ckpt.get("model_version"),
        "input_type": "hand_landmarks",
        "sign_ids": sign_ids,
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm,
        "confidence_bins": bins,
        "predictions": predictions,
    }
    out = ML_ROOT / "exports" / "eval_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_path = ROOT / "docs" / "VALIDATION_REPORT.md"
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    auto_block = _build_auto_block(ckpt, sign_ids, accuracy, report, cm_array, bins)
    report_path.write_text(_splice(existing, auto_block), encoding="utf-8")
    print(f"Wrote {out} accuracy={accuracy:.4f}")


if __name__ == "__main__":
    main()
