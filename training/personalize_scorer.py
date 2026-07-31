"""Calibrate a global scorer for unseen users with fixed HP or per-user HPO."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import optuna
import torch

from datasets.flickr_aes import build_client_loaders
from losses.scorer import personalized_scorer_objective
from models.scorer import FusionScorer, set_calibration_mask
from training.common import balanced_support_loader, class_weights_from_loader, evaluate_scorer
from utils.checkpoints import load_state_dict, save_checkpoint
from utils.reproducibility import resolve_device, seed_everything


@dataclass(frozen=True)
class CalibrationConfig:
    learning_rate: float
    lambda_reg: float
    lambda_pair: float
    lambda_var: float
    weight_decay: float
    gradient_clip: float
    calibration_epochs: int


def fixed_hp_config(support_size: int) -> CalibrationConfig:
    if support_size <= 20:
        return CalibrationConfig(5e-5, 1.0, 0.0, 0.02, 1e-5, 1.0, 30)
    return CalibrationConfig(1e-4, 0.85, 0.15, 0.02, 1e-5, 1.0, 60)


def _client_seed(client_id: str, base_seed: int) -> int:
    if client_id.isdigit():
        return base_seed + int(client_id)
    digest = hashlib.sha256(client_id.encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:4], "little")


def load_global_scorer(path: Path, device: torch.device) -> FusionScorer:
    model = FusionScorer().to(device)
    return load_state_dict(model, path, device)  # type: ignore[return-value]


def calibrate(
    global_scorer: FusionScorer,
    support_loader,
    val_loader,
    config: CalibrationConfig,
    device: torch.device,
    *,
    patience: int,
    max_batches: int | None,
) -> tuple[FusionScorer, dict]:
    scorer = copy.deepcopy(global_scorer).to(device)
    trainable = set_calibration_mask(scorer, len(support_loader.dataset))
    optimizer = torch.optim.Adam(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    class_weights = class_weights_from_loader(support_loader, max_batches=max_batches)
    best_state = copy.deepcopy(scorer.state_dict())
    best_srcc = -float("inf")
    stale_epochs = 0
    history = {"train_loss": [], "val_srcc": [], "val_mse": []}

    for _ in range(config.calibration_epochs):
        scorer.train()
        running_loss = 0.0
        samples = 0
        for batch_index, batch in enumerate(support_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            color = batch["color_features"].to(device).float()
            semantic = batch["semantic_features"].to(device).float()
            targets = batch["score"].to(device).float().view(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            predictions = scorer(color, semantic)
            loss, _ = personalized_scorer_objective(
                predictions,
                targets,
                lambda_reg=config.lambda_reg,
                lambda_pair=config.lambda_pair,
                lambda_var=config.lambda_var,
                class_weights=class_weights,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip)
            optimizer.step()
            running_loss += float(loss.item()) * len(targets)
            samples += len(targets)

        validation = evaluate_scorer(scorer, val_loader, device, max_batches=max_batches)
        history["train_loss"].append(running_loss / max(samples, 1))
        history["val_srcc"].append(validation["srcc"])
        history["val_mse"].append(validation["mse"])
        selection_value = validation["srcc"]
        if selection_value is not None and selection_value > best_srcc:
            best_srcc = float(selection_value)
            best_state = copy.deepcopy(scorer.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    scorer.load_state_dict(best_state)
    return scorer, {"best_val_srcc": None if best_srcc == -float("inf") else best_srcc, **history}


def hpo_config(
    trial: optuna.Trial,
    support_size: int,
    *,
    epochs_override: int | None,
) -> CalibrationConfig:
    lambda_pair = trial.suggest_float("lambda_pair", 0.0, 0.30)
    if epochs_override is not None:
        epochs = epochs_override
    elif support_size <= 20:
        epochs = trial.suggest_int("calibration_epochs", 20, 40)
    else:
        epochs = trial.suggest_int("calibration_epochs", 30, 80)
    return CalibrationConfig(
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        lambda_reg=max(0.60, 1.0 - lambda_pair),
        lambda_pair=lambda_pair,
        lambda_var=trial.suggest_float("lambda_var", 0.0, 0.05),
        weight_decay=trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
        gradient_clip=trial.suggest_float("gradient_clip", 0.5, 2.0),
        calibration_epochs=epochs,
    )


def personalize_client(
    args: argparse.Namespace,
    global_scorer: FusionScorer,
    client_id: str,
    support_size: int,
    device: torch.device,
) -> dict:
    loaders = build_client_loaders(
        args.data_root,
        client_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        load_precomputed_features=True,
        load_images=False,
    )
    seed = _client_seed(client_id, args.seed)
    support_loader = balanced_support_loader(
        loaders.train_fl.dataset,
        support_size,
        batch_size=args.batch_size,
        seed=seed,
    )

    if args.mode == "hpo":
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial: optuna.Trial) -> float:
            trial_seed = seed + trial.number
            seed_everything(trial_seed, args.deterministic)
            config = hpo_config(trial, support_size, epochs_override=args.epochs)
            _, history = calibrate(
                global_scorer,
                support_loader,
                loaders.val,
                config,
                device,
                patience=args.patience,
                max_batches=args.max_batches,
            )
            trial.set_user_attr("config", asdict(config))
            return -1.0 if history["best_val_srcc"] is None else float(history["best_val_srcc"])

        study.optimize(objective, n_trials=args.trials)
        config = CalibrationConfig(**study.best_trial.user_attrs["config"])
        trial_records = study.trials_dataframe()
    else:
        config = fixed_hp_config(support_size)
        if args.epochs is not None:
            config = CalibrationConfig(**{**asdict(config), "calibration_epochs": args.epochs})
        trial_records = None

    seed_everything(seed, args.deterministic)
    scorer, history = calibrate(
        global_scorer,
        support_loader,
        loaders.val,
        config,
        device,
        patience=args.patience,
        max_batches=args.max_batches,
    )
    test_metrics = evaluate_scorer(scorer, loaders.test, device, max_batches=args.max_batches)

    client_output = Path(args.output_dir) / f"{support_size}_samples"
    client_output.mkdir(parents=True, exist_ok=True)
    variant = "hpo" if args.mode == "hpo" else "fixed_hp"
    checkpoint_name = f"client_{client_id}_{variant}_personalized_scorer.pt"
    save_checkpoint(
        scorer,
        client_output / checkpoint_name,
        client_id=client_id,
        support_size=support_size,
        config=asdict(config),
    )
    if trial_records is not None:
        trial_records.to_csv(client_output / f"client_{client_id}_hpo_trials.csv", index=False)

    result = {
        "client_id": client_id,
        "support_size": support_size,
        "mode": args.mode,
        "config": asdict(config),
        "history": history,
        "test_metrics": test_metrics,
        "checkpoint": checkpoint_name,
    }
    (client_output / f"client_{client_id}_{variant}_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def _resolve_clients(args: argparse.Namespace) -> list[str]:
    if args.client_ids:
        return [value.strip() for value in args.client_ids.split(",") if value.strip()]
    if args.split_file:
        split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
        clients = split.get("unseen_users", split.get("test_users", []))
        return [str(client_id) for client_id in clients]
    raise ValueError("Provide --client-ids or --split-file.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--global-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/personalized_scorers"))
    parser.add_argument("--client-ids")
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--support-sizes", default="10,100")
    parser.add_argument("--mode", choices=("fixed_hp", "hpo"), default="hpo")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, help="Optional smoke/debug epoch override.")
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--max-clients", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    global_scorer = load_global_scorer(args.global_checkpoint, device)
    clients = _resolve_clients(args)
    if args.max_clients is not None:
        clients = clients[: args.max_clients]
    support_sizes = [int(value) for value in args.support_sizes.split(",")]
    for client_id in clients:
        for support_size in support_sizes:
            result = personalize_client(args, global_scorer, client_id, support_size, device)
            print(
                f"Client {client_id} | {support_size}-shot | "
                f"test_srcc={result['test_metrics']['srcc']}"
            )


if __name__ == "__main__":
    main()
