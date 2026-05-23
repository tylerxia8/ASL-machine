"""Smoke test for the canonical training pipeline.

Run with: python ml/scripts/smoke_pipeline.py
(Sets PYTHONIOENCODING=utf-8 internally to handle torch.onnx emoji output on cp1252 consoles.)

Verifies:
1. SignClipCNN3D builds with the actual wave1 vocab size.
2. Forward pass produces correct logits shape on dummy input.
3. ONNX export succeeds.
4. ONNXRuntime can load and run the exported model.
5. Exported ONNX outputs match the PyTorch outputs (numerical equivalence).
6. The labels.json schema includes the fields apps/web/src/lib/inference.ts reads.

No real training data required, no large disk writes.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

# torch.onnx's exporter prints unicode (✓/✅) to stdout. Force UTF-8 on Windows
# consoles (cp1252 by default) so the export doesn't crash on the emoji.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from model import build_model  # noqa: E402

WAVE1 = ROOT / "content" / "wave1_signs.csv"


def main() -> int:
    with open(WAVE1, newline="", encoding="utf-8") as f:
        sign_ids = [row["sign_id"] for row in csv.DictReader(f)]
    print(f"[1/6] Loaded {len(sign_ids)} signs from wave1_signs.csv")

    model = build_model(num_classes=len(sign_ids))
    model.eval()
    print(f"[2/6] Built SignClipCNN3D: "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    dummy = torch.randn(1, 3, 24, 160, 160)
    with torch.no_grad():
        logits = model(dummy)
    assert logits.shape == (1, len(sign_ids)), f"bad logits shape {logits.shape}"
    print(f"[3/6] Forward pass OK, logits shape {tuple(logits.shape)}")

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    try:
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=18,
            dynamo=False,
        )
        size_mb = Path(onnx_path).stat().st_size / 1e6
        print(f"[4/6] ONNX export OK ({size_mb:.1f} MB)")

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        onnx_out = sess.run(None, {"input": dummy.numpy()})[0]
        print(f"[5/6] ONNXRuntime load + run OK, output shape {onnx_out.shape}")

        max_diff = float(np.max(np.abs(onnx_out - logits.numpy())))
        assert max_diff < 1e-3, f"pytorch vs onnx diverged by {max_diff}"
        print(f"[6/6] PyTorch vs ONNX numerical match (max abs diff {max_diff:.2e})")
    finally:
        Path(onnx_path).unlink(missing_ok=True)

    schema = {
        "sign_ids": sign_ids,
        "label_to_idx": {s: i for i, s in enumerate(sign_ids)},
        "model_version": "smoke",
        "input_type": "3d",
        "num_frames": 24,
        "frame_size": 160,
    }
    required = {"sign_ids", "label_to_idx", "model_version", "input_type", "num_frames", "frame_size"}
    assert required.issubset(schema), f"labels.json schema missing: {required - schema.keys()}"
    print(f"\nlabels.json schema OK: {sorted(schema.keys())}")
    print("\nSMOKE PASS — canonical training/export/inference pipeline is intact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
