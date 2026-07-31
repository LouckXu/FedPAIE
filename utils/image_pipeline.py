"""Canonical image preprocessing and scorer feature extraction."""

from __future__ import annotations

import kornia.color as kcolor
import torch
from torch import nn
from torchvision import models

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize RGB images in ``[0, 1]`` for MobileNetV3."""
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected [B,3,H,W], got {tuple(images.shape)}")
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def imagenet_denormalize(images: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization and clamp RGB values to ``[0, 1]``."""
    squeezed = images.ndim == 3
    if squeezed:
        images = images.unsqueeze(0)
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected [B,3,H,W], got {tuple(images.shape)}")
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    restored = (images * std + mean).clamp(0.0, 1.0)
    return restored.squeeze(0) if squeezed else restored


class HSVLabFeatureExtractor(nn.Module):
    """Return the 24-D HSV and CIE Lab statistics used in the paper."""

    output_dim = 24

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected [B,3,H,W], got {tuple(images.shape)}")

        images = images.clamp(0.0, 1.0)
        hsv = kcolor.rgb_to_hsv(images).clone()
        hsv[:, :1] = hsv[:, :1] / (2.0 * torch.pi)

        lab = kcolor.rgb_to_lab(images).clone()
        lab[:, :1] = lab[:, :1] / 100.0
        lab[:, 1:2] = lab[:, 1:2] / 128.0
        lab[:, 2:3] = lab[:, 2:3] / 128.0

        statistics: list[torch.Tensor] = []
        for color_space in (hsv, lab):
            for channel_index in range(3):
                channel = color_space[:, channel_index]
                statistics.extend(
                    (
                        channel.mean(dim=(1, 2)),
                        channel.std(dim=(1, 2), unbiased=False),
                        channel.amin(dim=(1, 2)),
                        channel.amax(dim=(1, 2)),
                    )
                )
        return torch.stack(statistics, dim=1).float()


class MobileNetV3FeatureExtractor(nn.Module):
    """Frozen ImageNet MobileNetV3-Large trunk returning a 960-D embedding."""

    output_dim = 960

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        network = models.mobilenet_v3_large(weights=weights)
        self.features = network.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):  # type: ignore[override]
        """Keep the frozen extractor in evaluation mode."""
        return super().train(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 3:
            images = images.unsqueeze(0)
        features = self.features(images)
        return self.pool(features).flatten(1)


def score_images(
    scorer: nn.Module,
    images: torch.Tensor,
    color_extractor: HSVLabFeatureExtractor,
    semantic_extractor: MobileNetV3FeatureExtractor,
) -> torch.Tensor:
    """Score RGB images while preserving gradients to the input images."""
    images = images.clamp(0.0, 1.0)
    color_features = color_extractor(images)
    semantic_features = semantic_extractor(imagenet_normalize(images))
    return scorer(color_features, semantic_features)
