"""From-scratch 3D CNN for sign clip classification. No pretrained weights.

Two variants:
- SignClipCNN3D (~3.57M params): the "default" capacity, intended for >5k clips.
- SignClipCNN3D_Small (~720K params): roughly 5× smaller, for low-data regimes
  (<2k clips) where the default variant mode-collapses. Picked via the
  `--model-size {default|small}` arg to ml/train.py.

Both produce the same input/output contract: (N, 3, 24, 160, 160) → (N, num_classes).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SignClipCNN3D(nn.Module):
    """Default capacity: 3 Conv3d blocks (32→64→128) + 256-unit FC head.

    ~3.57M params. Use when training data has ≥5k clips across ≥5 signers.
    """

    def __init__(self, num_classes: int, num_frames: int = 24, height: int = 160, width: int = 160):
        super().__init__()
        self.num_frames = num_frames
        self.stem = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            # AvgPool3d with explicit kernel rather than AdaptiveAvgPool3d so the
            # graph exports cleanly to ONNX (torch 2.12 dynamo exporter has no
            # decomposition for _adaptive_avg_pool3d). For a fixed 24x160x160
            # input the tensor reaching here is (B, 128, 12, 40, 40); this
            # kernel produces (B, 128, 4, 5, 5) — same output shape.
            nn.AvgPool3d(kernel_size=(3, 8, 8)),
        )
        flat = 128 * 4 * 5 * 5
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stem(x))


class SignClipCNN3D_Small(nn.Module):
    """Reduced capacity for low-data regimes (~720K params).

    Halves every channel count (32→16, 64→32, 128→64) and reduces the FC head
    to 128 units. Same input/output contract as SignClipCNN3D, same ONNX export
    path. Smaller model is significantly less prone to the mode-collapse seen
    on wave1-semlex-full-v5/v6 (~600/1200 train clips × 25 classes).
    """

    def __init__(self, num_classes: int, num_frames: int = 24, height: int = 160, width: int = 160):
        super().__init__()
        self.num_frames = num_frames
        self.stem = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(kernel_size=(3, 8, 8)),
        )
        flat = 64 * 4 * 5 * 5
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),  # slightly higher dropout to compensate for smaller capacity
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stem(x))


def build_model(num_classes: int, size: str = "default", **kwargs):
    """Build a from-scratch 3D CNN. Always kaiming_normal_ initialization, no
    checkpoint loading — Req 7 (no pretrained models).

    size: "default" (3.57M params) or "small" (~720K params, low-data regime).
    """
    if size == "small":
        model: nn.Module = SignClipCNN3D_Small(num_classes=num_classes, **kwargs)
    elif size == "default":
        model = SignClipCNN3D(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model size {size!r}; must be 'default' or 'small'")
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return model
