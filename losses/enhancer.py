"""Frozen-scorer-guided enhancer objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def personalized_enhancement_objective(
    enhanced_scores: torch.Tensor,
    original_scores: torch.Tensor,
    enhanced_images: torch.Tensor,
    original_images: torch.Tensor,
    perceptual_distance: torch.Tensor,
    *,
    lambda_pref: float,
    lambda_aes: float,
    lambda_l1: float,
    lambda_perc: float,
    lambda_gap: float,
    mu: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute Equation 17 with paper-aligned parameter names."""
    gain = enhanced_scores - original_scores
    preference = -functional.logsigmoid(gain).mean()
    aesthetic = -enhanced_scores.mean()
    fidelity_l1 = (enhanced_images - original_images).abs().mean()
    excess_gap = functional.relu(gain - mu).mean()
    total = (
        lambda_pref * preference
        + lambda_aes * aesthetic
        + lambda_l1 * fidelity_l1
        + lambda_perc * perceptual_distance
        + lambda_gap * excess_gap
    )
    return total, {
        "preference": preference,
        "aesthetic": aesthetic,
        "l1": fidelity_l1,
        "perceptual": perceptual_distance,
        "excess_gap": excess_gap,
        "gain": gain.mean(),
    }
