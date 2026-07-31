"""Scorer objectives using the notation from the FedPAIE paper."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def normalized_rating_class(ratings: torch.Tensor) -> torch.Tensor:
    """Map normalized ratings in ``[0,1]`` to integer classes 1 through 5."""
    if torch.any((ratings < -1e-6) | (ratings > 1.0 + 1e-6)):
        raise ValueError("Expected normalized ratings in [0,1].")
    return ratings.clamp(0.0, 1.0).mul(4.0).round().long().add(1)


def weighted_regression_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted rating regression ``L_reg`` (Equation 3)."""
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    squared_error = (predictions - targets).square()
    if class_weights is not None:
        classes = normalized_rating_class(targets)
        squared_error = squared_error * class_weights.to(predictions.device)[classes]
    return squared_error.mean()


def pairwise_ordering_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    minimum_separation: float = 0.1,
) -> torch.Tensor:
    """Smooth unordered-pair preference objective ``L_pair`` (Equation 10)."""
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    indices = torch.triu_indices(len(targets), len(targets), offset=1, device=targets.device)
    target_difference = targets[indices[0]] - targets[indices[1]]
    valid = target_difference.abs() > minimum_separation
    if not torch.any(valid):
        return predictions.sum() * 0.0
    direction = target_difference[valid].sign()
    prediction_difference = predictions[indices[0][valid]] - predictions[indices[1][valid]]
    return -functional.logsigmoid(direction * prediction_difference).mean()


def variance_preservation_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    retained_fraction: float = 0.7,
) -> torch.Tensor:
    """Prediction-collapse protection ``L_var`` (Equation 11)."""
    target_std = targets.view(-1).std(unbiased=False).detach()
    prediction_std = predictions.view(-1).std(unbiased=False)
    return functional.relu(retained_fraction * target_std - prediction_std)


def personalized_scorer_objective(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    lambda_reg: float,
    lambda_pair: float,
    lambda_var: float,
    class_weights: torch.Tensor | None = None,
    collapse_threshold: float = 0.01,
    collapse_attenuation: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Return the full scorer-calibration objective and its components."""
    regression = weighted_regression_loss(predictions, targets, class_weights)
    pairwise = pairwise_ordering_loss(predictions, targets)
    variance = variance_preservation_loss(predictions, targets)
    effective_lambda_pair = float(lambda_pair)
    if predictions.detach().std(unbiased=False).item() < collapse_threshold:
        effective_lambda_pair *= collapse_attenuation
    total = (
        lambda_reg * regression
        + effective_lambda_pair * pairwise
        + lambda_var * variance
    )
    return total, {
        "regression": regression,
        "pairwise": pairwise,
        "variance": variance,
        "effective_lambda_pair": effective_lambda_pair,
    }
