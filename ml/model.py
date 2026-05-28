"""From-scratch 3D CNN for sign clip classification. No pretrained weights.

Two variants:
- SignClipCNN3D (~3.57M params): the "default" capacity, intended for >5k clips.
- SignClipCNN3D_Small (~720K params): roughly 5× smaller, for low-data regimes
  (<2k clips) where the default variant mode-collapses. Picked via the
  `--model-size {default|small}` arg to ml/train.py.
- SignClipFrameCNN (~160K params): experimental frame-wise 2D CNN with temporal
  mean/delta pooling. Keeps the same browser input contract while testing
  whether a simpler temporal formulation generalizes better on Sem-Lex.
- SignClipFrameTCN (~220K params): experimental frame-wise 2D CNN followed by
  temporal Conv1d blocks, still trained from scratch.
- SignClipMotionTCN (~390K params): dual-stream frame encoder that models RGB
  appearance and per-frame motion before temporal Conv1d blocks.

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


class SignClipFrameCNN(nn.Module):
    """Frame-wise 2D CNN with temporal pooling for low-data experiments."""

    def __init__(self, num_classes: int, num_frames: int = 24, height: int = 160, width: int = 160):
        super().__init__()
        self.num_frames = num_frames
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(height // 8, width // 8)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(96 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        feat = self.frame_encoder(frames).reshape(b, t, -1)
        mean_feat = feat.mean(dim=1)
        delta_feat = torch.abs(feat[:, 1:, :] - feat[:, :-1, :]).mean(dim=1)
        return self.head(torch.cat([mean_feat, delta_feat], dim=1))


class SignClipFrameTCN(nn.Module):
    """Frame-wise 2D encoder plus temporal Conv1d blocks."""

    def __init__(self, num_classes: int, num_frames: int = 24, height: int = 160, width: int = 160):
        super().__init__()
        self.num_frames = num_frames
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(height // 8, width // 8)),
            nn.Flatten(),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        feat = self.frame_encoder(frames).reshape(b, t, -1).permute(0, 2, 1)
        temporal = self.temporal(feat)
        mean_feat = temporal.mean(dim=2)
        max_feat = temporal.amax(dim=2)
        return self.head(torch.cat([mean_feat, max_feat], dim=1))


class SignClipMotionTCN(nn.Module):
    """Dual-stream frame-wise appearance + motion encoder plus temporal Conv1d blocks.

    The model still trains from scratch on the same RGB clip tensor, but it gives
    the temporal stack an explicit frame-difference stream. That should help
    signs where movement carries most of the distinction while preserving the
    browser's existing (N, 3, 24, 160, 160) input contract.
    """

    def __init__(self, num_classes: int, num_frames: int = 24, height: int = 160, width: int = 160):
        super().__init__()
        self.num_frames = num_frames
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(height // 8, width // 8)),
            nn.Flatten(),
        )
        self.motion_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(height // 4, width // 4)),
            nn.Flatten(),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(144, 160, kernel_size=3, padding=1),
            nn.BatchNorm1d(160),
            nn.ReLU(inplace=True),
            nn.Conv1d(160, 160, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(160),
            nn.ReLU(inplace=True),
            nn.Conv1d(160, 160, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(160),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(160 * 2, 160),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(160, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        frames = x.permute(0, 2, 1, 3, 4)
        rgb = frames.reshape(b * t, c, h, w)
        rgb_feat = self.rgb_encoder(rgb).reshape(b, t, -1)

        diffs = torch.abs(frames[:, 1:, :, :, :] - frames[:, :-1, :, :, :])
        zero = torch.zeros_like(frames[:, :1, :, :, :])
        motion = torch.cat([zero, diffs], dim=1).reshape(b * t, c, h, w)
        motion_feat = self.motion_encoder(motion).reshape(b, t, -1)

        feat = torch.cat([rgb_feat, motion_feat], dim=2).permute(0, 2, 1)
        temporal = self.temporal(feat)
        mean_feat = temporal.mean(dim=2)
        max_feat = temporal.amax(dim=2)
        return self.head(torch.cat([mean_feat, max_feat], dim=1))


def build_model(num_classes: int, size: str = "default", **kwargs):
    """Build a from-scratch 3D CNN. Always kaiming_normal_ initialization, no
    checkpoint loading — Req 7 (no pretrained models).

    size: "default" (3.57M params), "small" (~720K params), "frame"
    (~160K params), "tcn" (~220K params), or "motion_tcn" (~390K params).
    """
    if size == "small":
        model: nn.Module = SignClipCNN3D_Small(num_classes=num_classes, **kwargs)
    elif size == "frame":
        model = SignClipFrameCNN(num_classes=num_classes, **kwargs)
    elif size == "tcn":
        model = SignClipFrameTCN(num_classes=num_classes, **kwargs)
    elif size == "motion_tcn":
        model = SignClipMotionTCN(num_classes=num_classes, **kwargs)
    elif size == "default":
        model = SignClipCNN3D(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(
            f"Unknown model size {size!r}; must be 'default', 'small', 'frame', 'tcn', or 'motion_tcn'"
        )
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return model
