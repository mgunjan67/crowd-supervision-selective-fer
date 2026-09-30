"""Swin-T baseline and the final-vector channel-gating variant.

The gate is applied once to torchvision Swin-T's pooled 768-dimensional output.
It is not inserted between Swin stages and it does not operate on a class token.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import Swin_T_Weights, swin_t


class ChannelGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.gate = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("ChannelGate expects a [batch, channels] tensor")
        return features * self.gate(features)


class SwinClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        gated: bool,
        pretrained: bool = True,
    ):
        super().__init__()
        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = swin_t(weights=weights)
        channels = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        self.channel_gate = ChannelGate(channels) if gated else nn.Identity()
        self.classifier = nn.Linear(channels, num_classes)
        self.variant = "gated" if gated else "swin"

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled_features = self.backbone(images)
        gated_features = self.channel_gate(pooled_features)
        return self.classifier(gated_features)


def build_model(variant: str, num_classes: int, *, pretrained: bool = True):
    if variant not in {"swin", "gated"}:
        raise ValueError(f"Unknown model variant: {variant}")
    return SwinClassifier(
        num_classes=num_classes,
        gated=(variant == "gated"),
        pretrained=pretrained,
    )
