from __future__ import annotations

import torch
import torch.nn as nn

from landmark_dataset import HAND_FEATURES, NUM_FRAMES


class HandLandmarkTCN(nn.Module):
    """Temporal classifier over MediaPipe hand landmarks.

    Input contract: (N, 132, 24), where each frame has two hand slots. Each hand
    slot stores wrist absolute xyz plus 21 wrist-relative xyz landmarks.
    """

    def __init__(self, num_classes: int, num_features: int = HAND_FEATURES, num_frames: int = NUM_FRAMES):
        super().__init__()
        self.num_features = num_features
        self.num_frames = num_frames
        self.temporal = nn.Sequential(
            nn.Conv1d(num_features, 160, kernel_size=3, padding=1),
            nn.BatchNorm1d(160),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Conv1d(160, 192, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Conv1d(192, 192, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(192 * 2, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(192, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.temporal(x)
        mean_feat = feat.mean(dim=2)
        max_feat = feat.amax(dim=2)
        return self.head(torch.cat([mean_feat, max_feat], dim=1))


def build_landmark_model(num_classes: int) -> nn.Module:
    model = HandLandmarkTCN(num_classes=num_classes)
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    return model
