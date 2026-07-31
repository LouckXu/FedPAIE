"""Shared scorer training utilities."""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch import nn
from torch.utils.data import DataLoader, Subset

from losses.scorer import normalized_rating_class


def class_weights_from_loader(
    loader: DataLoader,
    *,
    dominance_threshold: float = 0.60,
    minimum_weight: float = 0.5,
    maximum_weight: float = 3.0,
    max_batches: int | None = None,
) -> torch.Tensor | None:
    """Return normalized inverse-frequency weights only for imbalanced clients."""
    counts = Counter({rating: 0 for rating in range(1, 6)})
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        classes = normalized_rating_class(batch["score"].view(-1)).tolist()
        counts.update(classes)
    total = sum(counts.values())
    if total == 0 or max(counts.values()) / total <= dominance_threshold:
        return None

    present = [total / count for count in counts.values() if count > 0]
    mean_inverse = sum(present) / len(present)
    weights = torch.ones(6, dtype=torch.float32)
    for rating_class, count in counts.items():
        if count > 0:
            value = (total / count) / mean_inverse
            weights[rating_class] = min(maximum_weight, max(minimum_weight, value))
    return weights


def balanced_support_loader(
    dataset,
    support_size: int,
    *,
    batch_size: int,
    seed: int,
) -> DataLoader:
    """Select a deterministic, approximately class-balanced rated support set."""
    if support_size > len(dataset):
        raise ValueError(f"Requested {support_size} support ratings from {len(dataset)} samples.")
    groups: dict[int, list[int]] = {rating: [] for rating in range(1, 6)}
    for index, raw_score in enumerate(dataset.records["score"].tolist()):
        rating = int(round(float(raw_score)))
        if 1 <= rating <= 5:
            groups[rating].append(index)

    rng = random.Random(seed)
    for indices in groups.values():
        rng.shuffle(indices)
    selected: list[int] = []
    while len(selected) < support_size:
        progress = False
        for rating in range(1, 6):
            if groups[rating] and len(selected) < support_size:
                selected.append(groups[rating].pop())
                progress = True
        if not progress:
            break

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(dataset, selected),
        batch_size=min(batch_size, len(selected)),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )


def _correlation(function, targets: np.ndarray, predictions: np.ndarray) -> float | None:
    if len(targets) < 2 or np.std(targets) == 0 or np.std(predictions) == 0:
        return None
    return float(function(targets, predictions)[0])


@torch.no_grad()
def evaluate_scorer(
    model: nn.Module,
    loaders: DataLoader | list[DataLoader],
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> dict[str, float | None]:
    """Evaluate one client, or macro-average metrics across client loaders."""
    model.eval()
    if not isinstance(loaders, DataLoader):
        client_results = [
            evaluate_scorer(model, loader, device, max_batches=max_batches)
            for loader in loaders
        ]

        def macro_mean(key: str) -> float | None:
            values = [result[key] for result in client_results if result[key] is not None]
            return float(np.mean(values)) if values else None

        return {
            "mse": macro_mean("mse"),
            "plcc": macro_mean("plcc"),
            "srcc": macro_mean("srcc"),
            "prediction_std": macro_mean("prediction_std"),
            "samples": float(sum(result["samples"] or 0.0 for result in client_results)),
        }

    targets: list[float] = []
    predictions: list[float] = []
    for batch_index, batch in enumerate(loaders):
        if max_batches is not None and batch_index >= max_batches:
            break
        color = batch["color_features"].to(device).float()
        semantic = batch["semantic_features"].to(device).float()
        target = batch["score"].to(device).float().view(-1, 1)
        prediction = model(color, semantic)
        targets.extend(target.cpu().view(-1).tolist())
        predictions.extend(prediction.cpu().view(-1).tolist())

    target_array = np.asarray(targets, dtype=np.float64)
    prediction_array = np.asarray(predictions, dtype=np.float64)
    mse = float(np.mean((target_array - prediction_array) ** 2)) if len(target_array) else None
    return {
        "mse": mse,
        "plcc": _correlation(pearsonr, target_array, prediction_array),
        "srcc": _correlation(spearmanr, target_array, prediction_array),
        "prediction_std": float(np.std(prediction_array)) if len(prediction_array) else None,
        "samples": float(len(target_array)),
    }
