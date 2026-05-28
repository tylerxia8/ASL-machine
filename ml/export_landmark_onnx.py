from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from landmark_dataset import HAND_FEATURES, NUM_FRAMES
from landmark_model import build_landmark_model

ML_ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sign_ids = ckpt["sign_ids"]
    model = build_landmark_model(num_classes=len(sign_ids))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out_dir = ML_ROOT / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"
    dummy = torch.randn(1, HAND_FEATURES, NUM_FRAMES)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )

    labels = {
        "sign_ids": sign_ids,
        "label_to_idx": ckpt["label_to_idx"],
        "model_version": ckpt.get("model_version", "hand-landmarks"),
        "model_size": "hand_landmark_tcn",
        "input_type": "hand_landmarks",
        "num_frames": NUM_FRAMES,
        "n_features": HAND_FEATURES,
        "pretrained_detector": "mediapipe_hands",
    }
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")

    meta = {
        "model_version": ckpt.get("model_version", "hand-landmarks"),
        "model_size": "hand_landmark_tcn",
        "input_type": "hand_landmarks",
        "num_classes": len(sign_ids),
        "val_accuracy": ckpt.get("val_accuracy"),
        "val_distinct_preds": ckpt.get("val_distinct_preds"),
        "val_acc_history": ckpt.get("val_acc_history"),
        "n_features": HAND_FEATURES,
        "num_frames": NUM_FRAMES,
        "pretrained_detector": "mediapipe_hands",
        "input_shape": [1, HAND_FEATURES, NUM_FRAMES],
    }
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Exported {onnx_path} ({len(sign_ids)} classes)")


if __name__ == "__main__":
    main()
