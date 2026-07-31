"""Compact residual CLUT enhancer used by FedPAIE.

The implementation follows the factorized lookup-table formulation described
in the FedPAIE and CLUT-Net papers. It uses PyTorch ``grid_sample`` so no custom
CUDA extension is required.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as functional
from torch import nn
from torchvision.transforms import Normalize


def _conv_stage(
    input_channels: int,
    output_channels: int,
    *,
    normalization: bool,
) -> list[nn.Module]:
    layers: list[nn.Module] = [
        nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1),
        nn.LeakyReLU(0.2),
    ]
    if normalization:
        layers.append(nn.InstanceNorm2d(output_channels, affine=True))
    return layers


class Backbone(nn.Module):
    """Lightweight coefficient-prediction backbone."""

    last_channel = 128

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            *_conv_stage(3, 16, normalization=True),
            *_conv_stage(16, 32, normalization=True),
            *_conv_stage(32, 64, normalization=True),
            *_conv_stage(64, 128, normalization=True),
            *_conv_stage(128, 128, normalization=False),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


def _cube_to_lut(cubes: torch.Tensor) -> torch.Tensor:
    """Reorder factorized RGB cubes for trilinear sampling."""
    lookup_tables = torch.empty_like(cubes)
    lookup_tables[:, 0] = cubes[:, 0].permute(0, 2, 3, 1)
    lookup_tables[:, 1] = cubes[:, 1].permute(0, 2, 1, 3)
    lookup_tables[:, 2] = cubes[:, 2]
    return lookup_tables


class CLUT(nn.Module):
    """Compressed residual 3D-LUT bases."""

    def __init__(
        self,
        num_bases: int,
        dimension: int = 33,
        spatial_rank: int = -1,
        width_rank: int = -1,
    ) -> None:
        super().__init__()
        self.num = int(num_bases)
        self.dim = int(dimension)
        self.s = int(spatial_rank)
        self.w = int(width_rank)

        if self.s < 0 and self.w < 0:
            self.mode = "uncompressed"
            self.LUTs = nn.Parameter(torch.zeros(self.num, 3, self.dim, self.dim, self.dim))
        elif self.s >= 0 and self.w < 0:
            self.mode = "spatial"
            self.s_Layers = nn.Parameter(torch.rand(self.dim, self.s) / 5.0 - 0.1)
            self.LUTs = nn.Parameter(torch.zeros(self.s, self.num * 3 * self.dim * self.dim))
        elif self.s < 0 and self.w >= 0:
            self.mode = "width"
            self.w_Layers = nn.Parameter(torch.rand(self.w, self.dim * self.dim) / 5.0 - 0.1)
            self.LUTs = nn.Parameter(torch.zeros(self.num * 3 * self.dim, self.w))
        else:
            self.mode = "factorized"
            self.s_Layers = nn.Parameter(torch.rand(self.dim, self.s) / 5.0 - 0.1)
            self.w_Layers = nn.Parameter(torch.rand(self.w, self.dim * self.dim) / 5.0 - 0.1)
            self.LUTs = nn.Parameter(torch.zeros(self.s * self.num * 3, self.w))

    def reconstruct_luts(self) -> torch.Tensor:
        """Reconstruct all residual LUT bases as ``[M,3,D,D,D]``."""
        if self.mode == "uncompressed":
            return self.LUTs
        if self.mode == "spatial":
            cubes = self.s_Layers @ self.LUTs
            cubes = cubes.reshape(self.dim, self.num * 3, self.dim * self.dim)
        elif self.mode == "width":
            cubes = (self.LUTs @ self.w_Layers).reshape(
                self.num, 3, self.dim, self.dim, self.dim
            )
            return _cube_to_lut(cubes)
        else:
            cores = self.LUTs @ self.w_Layers
            cores = cores.reshape(self.s, self.num * 3 * self.dim * self.dim)
            cubes = self.s_Layers @ cores
            cubes = cubes.reshape(self.dim, self.num * 3, self.dim * self.dim)

        cubes = cubes.permute(1, 0, 2).reshape(
            self.num, 3, self.dim, self.dim, self.dim
        )
        return _cube_to_lut(cubes)

    def forward(
        self,
        coefficients: torch.Tensor,
        regularizer: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | float]:
        bases = self.reconstruct_luts()
        penalty: torch.Tensor | float = 0.0 if regularizer is None else regularizer(bases)
        fused = coefficients @ bases.reshape(self.num, -1)
        fused = fused.reshape(-1, 3, self.dim, self.dim, self.dim)
        return fused, penalty


class TrilinearInterpolation(nn.Module):
    """Apply one image-conditioned LUT to each image in a batch."""

    def forward(self, lookup_tables: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        if lookup_tables.ndim != 5 or images.ndim != 4:
            raise ValueError("Expected LUTs [B,3,D,D,D] and images [B,3,H,W].")
        if lookup_tables.shape[0] != images.shape[0]:
            raise ValueError("LUT and image batch sizes must match.")

        grid = images.permute(0, 2, 3, 1).unsqueeze(1).mul(2.0).sub(1.0)
        channels = []
        for channel_index in range(3):
            sampled = functional.grid_sample(
                lookup_tables[:, channel_index : channel_index + 1],
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            channels.append(sampled[:, 0, 0])
        return torch.stack(channels, dim=1)


class TVMN(nn.Module):
    """Smoothness and monotonicity regularizer for reconstructed LUT bases."""

    def __init__(
        self,
        dimension: int = 33,
        lambda_smooth: float = 1e-4,
        lambda_monotonicity: float = 10.0,
    ) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_monotonicity = float(lambda_monotonicity)

    def forward(self, lookup_tables: torch.Tensor) -> torch.Tensor:
        differences = (
            lookup_tables[..., :-1] - lookup_tables[..., 1:],
            lookup_tables[..., :-1, :] - lookup_tables[..., 1:, :],
            lookup_tables[..., :-1, :, :] - lookup_tables[..., 1:, :, :],
        )
        smoothness = sum(diff.square().mean() for diff in differences)
        monotonicity = sum(functional.relu(diff).square().mean() for diff in differences)
        return self.lambda_smooth * smoothness + self.lambda_monotonicity * monotonicity


class CLUTNet(nn.Module):
    """Image-adaptive residual color-grading network."""

    def __init__(
        self,
        architecture: str = "20+05+20",
        dimension: int = 33,
        backbone: str | type[nn.Module] = "Backbone",
    ) -> None:
        super().__init__()
        parts = architecture.split("+")
        if len(parts) != 3:
            raise ValueError("architecture must have the form 'bases+spatial_rank+width_rank'.")
        num_bases, spatial_rank, width_rank = (int(part) for part in parts)

        registry: dict[str, type[nn.Module]] = {"Backbone": Backbone}
        backbone_type = registry[backbone] if isinstance(backbone, str) else backbone

        self.TrilinearInterpolation = TrilinearInterpolation()
        self.pre = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.backbone = backbone_type()
        self.classifier = nn.Sequential(
            nn.Conv2d(self.backbone.last_channel, 128, 1),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2, inplace=True),
            nn.Conv2d(128, num_bases, 1),
        )
        self.CLUTs = CLUT(num_bases, dimension, spatial_rank, width_rank)

    def fuse_basis_to_one(
        self,
        images: torch.Tensor,
        regularizer: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | float]:
        features = self.backbone(self.pre(images))
        coefficients = self.classifier(features)[:, :, 0, 0]
        return self.CLUTs(coefficients, regularizer)

    def forward(
        self,
        images: torch.Tensor,
        original_images: torch.Tensor | None = None,
        TVMN: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor | float]:
        original_images = images if original_images is None else original_images
        lookup_tables, regularization = self.fuse_basis_to_one(images, TVMN)
        residual = self.TrilinearInterpolation(lookup_tables, original_images)
        return {
            "fakes": original_images + residual,
            "3DLUT": lookup_tables,
            "tvmn_loss": regularization,
        }
