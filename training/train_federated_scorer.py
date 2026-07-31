"""Simulate Federated Aesthetic Preference Learning on Flickr-AES clients."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets.flickr_aes import build_client_loaders, list_client_ids
from losses.scorer import weighted_regression_loss
from models.scorer import FusionScorer
from training.common import class_weights_from_loader, evaluate_scorer
from utils.checkpoints import save_checkpoint
from utils.reproducibility import resolve_device, seed_everything


def _parse_ids(value: str | None) -> list[str] | None:
    return None if not value else [item.strip() for item in value.split(",") if item.strip()]


def _split_csv(data_root: Path, client_id: str, split: str) -> Path:
    return data_root / "split" / "flickr_dp4_split" / f"{client_id}_{split}.csv"


def _split_size(data_root: Path, client_id: str, split: str) -> int:
    return len(pd.read_csv(_split_csv(data_root, client_id, split), usecols=["score"]))


def create_open_world_split(
    data_root: Path,
    client_ids: list[str],
    *,
    unseen_users: int,
    seed: int,
    minimum_support_samples: int = 100,
    minimum_val_samples: int = 10,
    minimum_test_samples: int = 10,
) -> tuple[list[str], list[str]]:
    """Hold out entire eligible identities before federated optimization."""
    shuffled = list(client_ids)
    random.Random(seed).shuffle(shuffled)
    eligible = [
        client_id
        for client_id in shuffled
        if _split_size(data_root, client_id, "personalization") >= minimum_support_samples
        and _split_size(data_root, client_id, "val") >= minimum_val_samples
        and _split_size(data_root, client_id, "test") >= minimum_test_samples
    ]
    if len(eligible) < unseen_users:
        raise ValueError(
            f"Requested {unseen_users} unseen users, but only {len(eligible)} are eligible."
        )
    test_users = eligible[:unseen_users]
    train_users = [client_id for client_id in client_ids if client_id not in set(test_users)]
    return train_users, test_users


def filter_federated_clients(
    data_root: Path,
    client_ids: list[str],
    *,
    minimum_train_samples: int,
    require_label_diversity: bool,
) -> list[str]:
    """Apply the sample-count and rating-diversity controls from the paper."""
    eligible = []
    for client_id in client_ids:
        ratings = pd.read_csv(
            _split_csv(data_root, client_id, "train_fl"),
            usecols=["score"],
        )["score"]
        normalized = (ratings.astype(float) - 1.0) / 4.0
        diverse = normalized.nunique() >= 3 and float(normalized.std(ddof=0)) >= 0.12
        if len(ratings) >= minimum_train_samples and (diverse or not require_label_diversity):
            eligible.append(client_id)
    return eligible


def adaptive_local_epochs(sample_count: int) -> int:
    return 2 if sample_count < 500 else 1


def train_local_scorer(
    global_model: FusionScorer,
    loader,
    device: torch.device,
    *,
    learning_rate: float,
    weight_decay: float,
    local_epochs: int,
    gradient_clip: float,
    max_batches: int | None,
) -> tuple[OrderedDict[str, torch.Tensor], int, float]:
    """Optimize only ``L_reg`` and return parameters plus processed count ``m_k``."""
    local_model = copy.deepcopy(global_model).to(device).train()
    optimizer = torch.optim.AdamW(
        local_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    class_weights = class_weights_from_loader(
        loader,
        dominance_threshold=0.55,
        minimum_weight=0.5,
        maximum_weight=2.5,
        max_batches=max_batches,
    )
    total_loss = 0.0
    processed = 0

    for _ in range(local_epochs):
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            color = batch["color_features"].to(device).float()
            semantic = batch["semantic_features"].to(device).float()
            targets = batch["score"].to(device).float().view(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            predictions = local_model(color, semantic)
            loss = weighted_regression_loss(predictions, targets, class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), gradient_clip)
            optimizer.step()
            batch_size = len(targets)
            processed += batch_size
            total_loss += float(loss.item()) * batch_size

    state = OrderedDict(
        (key, value.detach().cpu()) for key, value in local_model.state_dict().items()
    )
    return state, processed, total_loss / max(processed, 1)


def square_root_fedavg(
    client_updates: list[tuple[OrderedDict[str, torch.Tensor], int]],
) -> OrderedDict[str, torch.Tensor]:
    """Aggregate client updates with ``alpha_k proportional to sqrt(m_k)``."""
    if not client_updates:
        raise ValueError("No client updates were produced.")
    raw_weights = [math.sqrt(max(processed, 1)) for _, processed in client_updates]
    normalizer = sum(raw_weights)
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in client_updates[0][0]:
        result[key] = sum(
            state[key] * (weight / normalizer)
            for (state, _), weight in zip(client_updates, raw_weights)
        )
    return result


def train_federated_scorer(args: argparse.Namespace) -> dict:
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    seed_everything(args.seed, args.deterministic)

    client_ids = _parse_ids(args.client_ids) or list_client_ids(data_root)
    train_users, unseen_users = create_open_world_split(
        data_root,
        client_ids,
        unseen_users=args.unseen_users,
        seed=args.seed,
    )
    federated_clients = filter_federated_clients(
        data_root,
        train_users,
        minimum_train_samples=args.min_train_samples,
        require_label_diversity=not args.no_label_diversity_filter,
    )
    if not federated_clients:
        raise ValueError("No clients remain after federated eligibility filtering.")

    requested_validation = _parse_ids(args.validation_client_ids)
    rng = random.Random(args.seed)
    validation_clients = requested_validation or rng.sample(
        federated_clients, min(args.validation_users, len(federated_clients))
    )

    split_payload = {
        "seed": args.seed,
        "train_users": train_users,
        "unseen_users": unseen_users,
        "federated_clients": federated_clients,
        "validation_clients": validation_clients,
    }
    (output_dir / "open_world_user_split.json").write_text(
        json.dumps(split_payload, indent=2), encoding="utf-8"
    )

    model = FusionScorer().to(device)
    history: list[dict] = []
    start_round = 0
    best_srcc = -float("inf")
    latest_path = output_dir / "checkpoint_latest.pt"
    if args.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_round = int(checkpoint.get("round", 0))
        history = list(checkpoint.get("history", []))
        best_srcc = float(checkpoint.get("best_srcc", best_srcc))

    for round_index in range(start_round + 1, args.rounds + 1):
        if args.clients_per_round and args.clients_per_round < len(federated_clients):
            selected = rng.sample(federated_clients, args.clients_per_round)
        else:
            selected = list(federated_clients)

        updates = []
        client_losses = {}
        for client_id in selected:
            loaders = build_client_loaders(
                data_root,
                client_id,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                load_precomputed_features=True,
                load_images=False,
            )
            sample_count = len(loaders.train_fl.dataset)
            state, processed, loss = train_local_scorer(
                model,
                loaders.train_fl,
                device,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                local_epochs=adaptive_local_epochs(sample_count),
                gradient_clip=args.gradient_clip,
                max_batches=args.max_batches_per_client,
            )
            updates.append((state, processed))
            client_losses[str(client_id)] = loss

        model.load_state_dict(square_root_fedavg(updates))
        val_loaders = [
            build_client_loaders(
                data_root,
                client_id,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                load_precomputed_features=True,
                load_images=False,
            ).val
            for client_id in validation_clients
        ]
        validation = evaluate_scorer(
            model,
            val_loaders,
            device,
            max_batches=args.max_validation_batches,
        )
        round_record = {
            "round": round_index,
            "selected_clients": selected,
            "mean_train_loss": float(np.mean(list(client_losses.values()))),
            "validation": validation,
        }
        history.append(round_record)
        current_srcc = validation["srcc"]
        if current_srcc is not None and current_srcc > best_srcc:
            best_srcc = float(current_srcc)
            save_checkpoint(
                model,
                output_dir / "checkpoint_best_srcc.pt",
                round=round_index,
                validation=validation,
            )
        save_checkpoint(
            model,
            latest_path,
            round=round_index,
            history=history,
            best_srcc=best_srcc,
        )
        print(
            f"Round {round_index:02d}/{args.rounds:02d} | "
            f"loss={round_record['mean_train_loss']:.6f} | "
            f"val_srcc={validation['srcc']}"
        )

    save_checkpoint(model, output_dir / "global_scorer_final.pt", history=history)
    return {"history": history, **split_payload}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/federated_scorer"))
    parser.add_argument("--client-ids", help="Optional comma-separated client subset.")
    parser.add_argument("--validation-client-ids", help="Optional fixed validation identities.")
    parser.add_argument("--unseen-users", type=int, default=37)
    parser.add_argument("--validation-users", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--min-train-samples", type=int, default=100)
    parser.add_argument(
        "--clients-per-round",
        type=int,
        default=0,
        help="0 uses every eligible client.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-label-diversity-filter", action="store_true")
    parser.add_argument("--max-batches-per-client", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    return parser


def main() -> None:
    train_federated_scorer(build_parser().parse_args())


if __name__ == "__main__":
    main()
