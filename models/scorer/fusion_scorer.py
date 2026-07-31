"""Lightweight Dual-Cue Aesthetic Scorer."""

from __future__ import annotations

import torch
from torch import nn


class FusionScorer(nn.Module):
    """Fuse a 24-D color descriptor with a 960-D semantic embedding."""

    def __init__(
        self,
        color_input_dim: int = 24,
        color_dim: int = 512,
        semantic_input_dim: int = 960,
        semantic_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.color_proj = nn.Sequential(
            nn.Linear(color_input_dim, color_dim),
            nn.LayerNorm(color_dim),
            nn.ReLU(),
        )
        self.semantic_proj = nn.Sequential(
            nn.Linear(semantic_input_dim, semantic_dim),
            nn.LayerNorm(semantic_dim),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(color_dim + semantic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.temperature = nn.Parameter(torch.tensor(2.0))

    def forward(
        self,
        color_features: torch.Tensor,
        semantic_features: torch.Tensor,
    ) -> torch.Tensor:
        color_latent = self.color_proj(color_features.float())
        semantic_latent = self.semantic_proj(semantic_features.float())
        logits = self.mlp(torch.cat((color_latent, semantic_latent), dim=1))
        temperature = self.temperature.clamp(1.0, 3.0)
        return torch.sigmoid(temperature * logits)


def set_calibration_mask(
    scorer: FusionScorer,
    support_size: int,
    *,
    projection_threshold: int = 20,
) -> list[nn.Parameter]:
    """Apply the support-dependent scorer mask from the paper.

    Up to ``projection_threshold`` ratings, only the fusion MLP and temperature
    are updated. Larger support sets also update both cue projections.
    """
    scorer.requires_grad_(False)
    modules: list[nn.Module] = [scorer.mlp]
    if support_size > projection_threshold:
        modules.extend((scorer.color_proj, scorer.semantic_proj))
    for module in modules:
        module.requires_grad_(True)
    scorer.temperature.requires_grad_(True)
    return [parameter for parameter in scorer.parameters() if parameter.requires_grad]
