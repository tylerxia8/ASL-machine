from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# torch.onnx's dynamo exporter prints unicode (✓/✅) to stdout. Force UTF-8 on
# Windows consoles (cp1252 by default) so export doesn't crash on the emoji.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

import torch

from model import build_model

ML_ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(ML_ROOT / "checkpoints" / "v1" / "best.pt"))
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sign_ids = ckpt["sign_ids"]
    model = build_model(num_classes=len(sign_ids))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy = torch.randn(1, 3, 24, 160, 160)
    out_dir = ML_ROOT / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"

    # dynamo=False uses the legacy TorchScript-based exporter which emits a single
    # self-contained .onnx file (the newer dynamo exporter splits weights into a
    # .onnx.data sidecar, which the sync-model script and browser loader don't expect).
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
        "model_version": ckpt.get("model_version", "v1"),
        "input_type": "3d",
        "num_frames": 24,
        "frame_size": 160,
    }
    with open(out_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    meta = {
        "model_version": ckpt.get("model_version", "v1"),
        "num_classes": len(sign_ids),
        "val_accuracy": ckpt.get("val_accuracy"),
        "pretrained": False,
        "input_shape": [1, 3, 24, 160, 160],
    }
    with open(out_dir / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Exported {onnx_path} ({len(sign_ids)} classes)")


if __name__ == "__main__":
    main()
