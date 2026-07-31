"""Run Frozen-Scorer-Guided Enhancer Adaptation for unseen Flickr-AES users."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import lpips
import numpy as np
import torch
from skimage.metrics import structural_similarity

from datasets.flickr_aes import build_client_loaders
from losses.enhancer import personalized_enhancement_objective
from models.scorer import FusionScorer
from utils.checkpoints import (
    clut_architecture,
    load_clut_model,
    load_state_dict,
    save_checkpoint,
)
from utils.image_pipeline import HSVLabFeatureExtractor, MobileNetV3FeatureExtractor, score_images
from utils.reproducibility import resolve_device, seed_everything


@dataclass(frozen=True)
class AdaptationConfig:
    learning_rate: float
    lambda_pref: float
    lambda_aes: float
    lambda_l1: float
    lambda_perc: float
    lambda_gap: float
    mu: float
    gradient_clip: float
    epochs: int = 40
    gamma_l1: float = 0.0
    gamma_perc: float = 0.0


def fixed_hp_config(
    validation_srcc: float,
    min_validation_srcc: float = 0.10,
) -> AdaptationConfig | None:
    """Paper fixed-HP configuration, including the SRCC-to-lambda rule."""
    if validation_srcc < min_validation_srcc:
        return None
    if validation_srcc < 0.20:
        lambda_pref = 0.01
    elif validation_srcc < 0.30:
        lambda_pref = 0.03
    elif validation_srcc < 0.40:
        lambda_pref = 0.05
    else:
        lambda_pref = 0.09
    return AdaptationConfig(3e-4, lambda_pref, 0.5, 0.1, 0.05, 3.0, 0.05, 1.0)


def shared_hpo_config(support_size: int) -> AdaptationConfig:
    """Selected shared-HPO settings reported in Supplementary Table 3."""
    if support_size == 10:
        values = (5.4485e-4, 0.0411, 0.5996, 0.1007, 0.0543, 0.5107, 0.1048, 1.9631)
    elif support_size == 100:
        values = (5.3948e-4, 0.0402, 0.5848, 0.1271, 0.0543, 0.7642, 0.2051, 1.6176)
    else:
        raise ValueError("Shared-HPO settings are reported only for 10- and 100-shot regimes.")
    learning_rate, pref, aes, l1_value, perc, gap, mu, clip = values
    return AdaptationConfig(
        learning_rate,
        pref,
        aes,
        l1_value,
        perc,
        gap,
        mu,
        clip,
        gamma_l1=l1_value,
        gamma_perc=perc,
    )


def select_adaptation_config(
    preset: str,
    support_size: int,
    validation_srcc: float,
    min_validation_srcc: float = 0.10,
) -> AdaptationConfig | None:
    """Apply the shared scorer-eligibility rule before selecting enhancer HPs."""
    if validation_srcc < min_validation_srcc:
        return None
    if preset == "fixed_hp":
        return fixed_hp_config(validation_srcc, min_validation_srcc)
    if preset == "shared_hpo":
        return shared_hpo_config(support_size)
    raise ValueError(f"Unsupported enhancer preset: {preset}")


def _scorer_root(path: Path, support_size: int) -> Path:
    nested = path / f"{support_size}_samples"
    return nested if nested.is_dir() else path


def _scorer_checkpoint(root: Path, client_id: str, mode: str) -> Path:
    candidates = (
        [
            root / f"client_{client_id}_hpo_personalized_scorer.pt",
            root / f"client_{client_id}_optuna_personalized_scorer.pt",
        ]
        if mode == "hpo"
        else [
            root / f"client_{client_id}_fixed_hp_personalized_scorer.pt",
            root / f"client_{client_id}_personalized_scorer.pt",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No {mode} scorer checkpoint found for client {client_id} in {root}")


def _validation_srcc(root: Path, client_id: str, mode: str) -> float:
    names = (
        [f"client_{client_id}_hpo_results.json", f"client_{client_id}_optuna_results.json"]
        if mode == "hpo"
        else [f"client_{client_id}_fixed_hp_results.json", f"client_{client_id}_results.json"]
    )
    for name in names:
        path = root / name
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("best_val_srcc") is not None:
            return float(result["best_val_srcc"])
        values = result.get("history", {}).get("val_srcc", [])
        valid = [float(value) for value in values if value is not None]
        if valid:
            return max(valid)
    raise FileNotFoundError(f"No validation SRCC record found for client {client_id} in {root}")


def _load_scorer(path: Path, device: torch.device) -> FusionScorer:
    model = FusionScorer().to(device)
    load_state_dict(model, path, device)
    model.requires_grad_(False).eval()
    return model


def _freeze_enhancer_bases(model) -> list[torch.nn.Parameter]:
    model.requires_grad_(False)
    model.backbone.requires_grad_(True)
    model.classifier.requires_grad_(True)
    model.CLUTs.eval()
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _original_scores(
    cache: dict[str, float],
    image_names: list[str],
    images: torch.Tensor,
    scorer,
    color_extractor,
    semantic_extractor,
) -> torch.Tensor:
    missing = [index for index, name in enumerate(image_names) if name not in cache]
    if missing:
        with torch.no_grad():
            computed = score_images(
                scorer,
                images[missing],
                color_extractor,
                semantic_extractor,
            ).view(-1)
        for index, value in zip(missing, computed.tolist()):
            cache[image_names[index]] = float(value)
    return images.new_tensor([cache[name] for name in image_names]).view(-1, 1)


@torch.no_grad()
def _validate(
    enhancer,
    loader,
    scorer,
    color_extractor,
    semantic_extractor,
    perceptual_model,
    score_cache: dict[str, float],
    config: AdaptationConfig,
    device: torch.device,
    max_batches: int | None,
) -> dict[str, float]:
    enhancer.eval()
    gains: list[float] = []
    l1_values: list[float] = []
    perceptual_values: list[float] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device).float()
        names = list(batch["image_name"])
        enhanced = enhancer(images, images)["fakes"].clamp(0.0, 1.0)
        original_scores = _original_scores(
            score_cache, names, images, scorer, color_extractor, semantic_extractor
        )
        enhanced_scores = score_images(scorer, enhanced, color_extractor, semantic_extractor)
        gains.extend((enhanced_scores - original_scores).view(-1).tolist())
        l1_values.append(float((enhanced - images).abs().mean().item()))
        perceptual_values.append(
            float(perceptual_model(enhanced * 2 - 1, images * 2 - 1).mean().item())
        )
    mean_gain = float(np.mean(gains)) if gains else -float("inf")
    mean_l1 = float(np.mean(l1_values)) if l1_values else 0.0
    mean_perceptual = float(np.mean(perceptual_values)) if perceptual_values else 0.0
    return {
        "gain": mean_gain,
        "l1": mean_l1,
        "lpips": mean_perceptual,
        "selection_criterion": (
            mean_gain - config.gamma_l1 * mean_l1 - config.gamma_perc * mean_perceptual
        ),
    }


@torch.no_grad()
def _test_metrics(
    enhancer,
    loader,
    scorer,
    color_extractor,
    semantic_extractor,
    perceptual_model,
    score_cache: dict[str, float],
    device: torch.device,
    max_batches: int | None,
) -> dict[str, float | None]:
    enhancer.eval()
    scores: list[float] = []
    gains: list[float] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device).float()
        names = list(batch["image_name"])
        enhanced = enhancer(images, images)["fakes"].clamp(0.0, 1.0)
        originals = _original_scores(
            score_cache, names, images, scorer, color_extractor, semantic_extractor
        )
        enhanced_scores = score_images(scorer, enhanced, color_extractor, semantic_extractor)
        scores.extend(enhanced_scores.view(-1).tolist())
        gains.extend((enhanced_scores - originals).view(-1).tolist())
        mse = (enhanced - images).square().flatten(1).mean(1).clamp_min(1e-12)
        psnr_values.extend((10.0 * torch.log10(1.0 / mse)).tolist())
        lpips_values.extend(
            perceptual_model(enhanced * 2 - 1, images * 2 - 1).view(-1).tolist()
        )
        for original, output in zip(images, enhanced):
            original_np = original.permute(1, 2, 0).cpu().numpy()
            output_np = output.permute(1, 2, 0).cpu().numpy()
            ssim_values.append(
                float(structural_similarity(original_np, output_np, channel_axis=2, data_range=1.0))
            )
    return {
        "predicted_score": float(np.mean(scores)) if scores else None,
        "predicted_gain": float(np.mean(gains)) if gains else None,
        "input_psnr": float(np.mean(psnr_values)) if psnr_values else None,
        "input_ssim": float(np.mean(ssim_values)) if ssim_values else None,
        "input_lpips": float(np.mean(lpips_values)) if lpips_values else None,
    }


def adapt_client(
    args: argparse.Namespace,
    client_id: str,
    device: torch.device,
    color_extractor,
    semantic_extractor,
    perceptual_model,
) -> dict:
    scorer_root = _scorer_root(Path(args.scorer_dir), args.support_size)
    scorer_path = _scorer_checkpoint(scorer_root, client_id, args.scorer_mode)
    validation_srcc = (
        args.validation_srcc
        if args.validation_srcc is not None
        else _validation_srcc(scorer_root, client_id, args.scorer_mode)
    )
    config = select_adaptation_config(
        args.preset,
        args.support_size,
        validation_srcc,
        args.min_validation_srcc,
    )
    if config is None:
        return {
            "client_id": client_id,
            "skipped": True,
            "validation_srcc": validation_srcc,
            "min_validation_srcc": args.min_validation_srcc,
        }
    if args.epochs is not None:
        config = AdaptationConfig(**{**asdict(config), "epochs": args.epochs})

    seed_everything(args.seed)
    scorer = _load_scorer(scorer_path, device)
    enhancer = load_clut_model(
        args.global_enhancer_checkpoint,
        device,
        architecture=args.architecture,
        allow_legacy_pickle=args.allow_legacy_checkpoint,
    )
    trainable = _freeze_enhancer_bases(enhancer)
    optimizer = torch.optim.Adam(trainable, lr=config.learning_rate)
    loaders = build_client_loaders(
        args.data_root,
        client_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize_images=False,
        load_precomputed_features=False,
        load_images=True,
    )

    score_cache: dict[str, float] = {}
    best_state = copy.deepcopy(enhancer.state_dict())
    best_selection = -float("inf")
    history = []
    for epoch in range(1, config.epochs + 1):
        enhancer.backbone.train()
        enhancer.classifier.train()
        running_loss = 0.0
        samples = 0
        for batch_index, batch in enumerate(loaders.personalization):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            images = batch["image"].to(device).float()
            names = list(batch["image_name"])
            optimizer.zero_grad(set_to_none=True)
            enhanced = enhancer(images, images)["fakes"].clamp(0.0, 1.0)
            original_scores = _original_scores(
                score_cache, names, images, scorer, color_extractor, semantic_extractor
            )
            enhanced_scores = score_images(scorer, enhanced, color_extractor, semantic_extractor)
            perceptual_distance = perceptual_model(
                enhanced * 2 - 1, images * 2 - 1
            ).mean()
            loss, _ = personalized_enhancement_objective(
                enhanced_scores,
                original_scores,
                enhanced,
                images,
                perceptual_distance,
                lambda_pref=config.lambda_pref,
                lambda_aes=config.lambda_aes,
                lambda_l1=config.lambda_l1,
                lambda_perc=config.lambda_perc,
                lambda_gap=config.lambda_gap,
                mu=config.mu,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip)
            optimizer.step()
            running_loss += float(loss.item()) * len(images)
            samples += len(images)

        validation = _validate(
            enhancer,
            loaders.val,
            scorer,
            color_extractor,
            semantic_extractor,
            perceptual_model,
            score_cache,
            config,
            device,
            args.max_batches,
        )
        history.append(
            {"epoch": epoch, "train_loss": running_loss / max(samples, 1), **validation}
        )
        if validation["selection_criterion"] > best_selection:
            best_selection = validation["selection_criterion"]
            best_state = copy.deepcopy(enhancer.state_dict())

    enhancer.load_state_dict(best_state)
    metrics = _test_metrics(
        enhancer,
        loaders.test,
        scorer,
        color_extractor,
        semantic_extractor,
        perceptual_model,
        score_cache,
        device,
        args.max_batches,
    )
    client_output = Path(args.output_dir) / f"client_{client_id}"
    checkpoint_name = f"client_{client_id}_personalized_enhancer.pt"
    save_checkpoint(
        enhancer,
        client_output / checkpoint_name,
        client_id=client_id,
        architecture=clut_architecture(enhancer),
    )
    result = {
        "client_id": client_id,
        "support_size": args.support_size,
        "preset": args.preset,
        "validation_srcc": validation_srcc,
        "min_validation_srcc": args.min_validation_srcc,
        "config": asdict(config),
        "best_selection_criterion": best_selection,
        "history": history,
        "test_metrics": metrics,
        "checkpoint": checkpoint_name,
    }
    (client_output / "adaptation_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--global-enhancer-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--architecture",
        help="Optional override for legacy state-dict checkpoints without metadata.",
    )
    parser.add_argument("--scorer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/personalized_enhancers"))
    parser.add_argument("--client-ids", required=True, help="Comma-separated unseen client IDs.")
    parser.add_argument("--support-size", type=int, choices=(10, 100), required=True)
    parser.add_argument("--scorer-mode", choices=("fixed_hp", "hpo"), default="hpo")
    parser.add_argument("--preset", choices=("fixed_hp", "shared_hpo"), default="shared_hpo")
    parser.add_argument(
        "--validation-srcc",
        type=float,
        help="Optional single-value override for smoke tests.",
    )
    parser.add_argument(
        "--min-validation-srcc",
        type=float,
        default=0.10,
        help="Minimum personalized-scorer validation SRCC required for enhancer adaptation.",
    )
    parser.add_argument(
        "--allow-legacy-checkpoint",
        action="store_true",
        help="Allow trusted legacy checkpoints that serialize a full Python model.",
    )
    parser.add_argument("--epochs", type=int, help="Optional smoke/debug epoch override.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--max-clients", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    color_extractor = HSVLabFeatureExtractor().to(device).eval()
    semantic_extractor = MobileNetV3FeatureExtractor().to(device).eval()
    perceptual_model = lpips.LPIPS(net="alex").to(device).eval()
    perceptual_model.requires_grad_(False)
    client_ids = [value.strip() for value in args.client_ids.split(",") if value.strip()]
    if args.max_clients is not None:
        client_ids = client_ids[: args.max_clients]
    results = []
    for client_id in client_ids:
        result = adapt_client(
            args,
            client_id,
            device,
            color_extractor,
            semantic_extractor,
            perceptual_model,
        )
        results.append(result)
        print(f"Client {client_id}: {result.get('test_metrics', result)}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "adaptation_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
