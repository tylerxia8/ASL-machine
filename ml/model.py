"""From-scratch 3D CNN for sign clip classification. No pretrained weights."""
from __future__ import annotations

import torch
import torch.nn as nn


class SignClipCNN3D(nn.Module):
    """Small 3D CNN: input (N, C, T, H, W). Random initialization only."""

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
        # Always (N, C, T, H, W). The dataset's __getitem__ and the browser's
        # framesToTensor both produce this layout, so no runtime permute needed.
        return self.head(self.stem(x))


def build_model(num_classes: int, **kwargs) -> SignClipCNN3D:
    model = SignClipCNN3D(num_classes=num_classes, **kwargs)
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return model
