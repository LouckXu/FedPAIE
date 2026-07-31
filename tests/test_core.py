from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from losses.enhancer import personalized_enhancement_objective
from losses.scorer import personalized_scorer_objective, weighted_regression_loss
from models.clut_net import CLUTNet
from models.scorer import FusionScorer, set_calibration_mask
from training.adapt_enhancer import select_adaptation_config
from training.common import evaluate_scorer
from utils.checkpoints import clut_architecture, load_clut_model, save_checkpoint


def test_paper_model_sizes() -> None:
    enhancer = CLUTNet()
    scorer = FusionScorer()
    assert sum(parameter.numel() for parameter in enhancer.parameters()) == 292_541
    assert sum(parameter.numel() for parameter in scorer.parameters()) == 787_202


def test_enhancer_forward_and_gradient() -> None:
    model = CLUTNet()
    images = torch.rand(2, 3, 32, 32)
    output = model(images, images)["fakes"]
    assert output.shape == images.shape
    output.mean().backward()
    assert model.classifier[-1].weight.grad is not None


def test_clut_architecture_round_trip(tmp_path) -> None:
    model = CLUTNet("03+02+04")
    checkpoint = tmp_path / "custom.pt"
    save_checkpoint(model, checkpoint, architecture=clut_architecture(model))
    restored = load_clut_model(checkpoint, "cpu")
    assert clut_architecture(restored) == "03+02+04"


def test_scorer_output_and_support_mask() -> None:
    scorer = FusionScorer()
    scores = scorer(torch.rand(3, 24), torch.rand(3, 960))
    assert scores.shape == (3, 1)
    assert torch.all((scores >= 0.0) & (scores <= 1.0))

    small_support = set_calibration_mask(scorer, 10)
    assert sum(parameter.numel() for parameter in small_support) == 526_850
    large_support = set_calibration_mask(scorer, 100)
    assert sum(parameter.numel() for parameter in large_support) == 787_202


def test_objectives_are_finite() -> None:
    predictions = torch.tensor([[0.2], [0.7], [0.6]])
    targets = torch.tensor([[0.1], [0.8], [0.5]])
    weights = torch.ones(6)
    regression = weighted_regression_loss(predictions, targets, weights)
    calibration, _ = personalized_scorer_objective(
        predictions,
        targets,
        lambda_reg=1.0,
        lambda_pair=0.1,
        lambda_var=0.1,
        class_weights=weights,
    )
    images = torch.rand(3, 3, 8, 8)
    enhancement, _ = personalized_enhancement_objective(
        predictions + 0.02,
        predictions,
        images,
        images,
        torch.tensor(0.0),
        lambda_pref=0.04,
        lambda_aes=0.6,
        lambda_l1=0.1,
        lambda_perc=0.05,
        lambda_gap=0.7,
        mu=0.1,
    )
    assert all(torch.isfinite(value) for value in (regression, calibration, enhancement))


def test_enhancer_presets_share_validation_srcc_eligibility() -> None:
    for preset in ("fixed_hp", "shared_hpo"):
        for support_size in (10, 100):
            assert select_adaptation_config(preset, support_size, 0.099) is None
            assert select_adaptation_config(preset, support_size, 0.10) is not None
            assert select_adaptation_config(preset, support_size, 0.099, 0.05) is not None


def test_validation_metrics_are_macro_averaged() -> None:
    class FirstFeatureScorer(nn.Module):
        def forward(self, color_features, semantic_features):
            del semantic_features
            return color_features[:, :1]

    def loader(predictions, targets):
        samples = [
            {
                "color_features": torch.tensor([prediction] + [0.0] * 23),
                "semantic_features": torch.zeros(960),
                "score": torch.tensor(target),
            }
            for prediction, target in zip(predictions, targets)
        ]
        return DataLoader(samples, batch_size=len(samples))

    result = evaluate_scorer(
        FirstFeatureScorer(),
        [loader([0.0, 1.0], [0.0, 1.0]), loader([1.0, 0.0], [0.0, 1.0])],
        torch.device("cpu"),
    )
    assert result["mse"] == 0.5
    assert result["srcc"] == 0.0
    assert result["samples"] == 4.0
