"""Train the Generic Enhancement Prior on paired MIT-Adobe FiveK images."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import lpips
import torch
from torch.utils.data import DataLoader

from datasets.fivek import FiveKPairedDataset
from models.clut_net import CLUTNet
from utils.checkpoints import load_state_dict, save_checkpoint
from utils.reproducibility import resolve_device, seed_everything


@torch.no_grad()
def evaluate(
    model: CLUTNet,
    loader: DataLoader,
    perceptual_model,
    device: torch.device,
    *,
    lambda_perc: float,
    max_batches: int | None,
) -> float:
    model.eval()
    total = 0.0
    samples = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        outputs = model(inputs, inputs)["fakes"]
        l1_loss = (outputs - targets).abs().mean()
        perceptual = perceptual_model(outputs * 2 - 1, targets * 2 - 1).mean()
        loss = l1_loss + lambda_perc * perceptual
        total += float(loss.item()) * len(inputs)
        samples += len(inputs)
    return total / max(samples, 1)


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed, args.deterministic)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = FiveKPairedDataset(
        args.input_dir,
        args.target_dir,
        image_size=args.image_size,
        manifest=args.train_manifest,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = None
    if bool(args.val_input_dir) != bool(args.val_target_dir):
        raise ValueError("Provide both --val-input-dir and --val-target-dir, or neither.")
    if args.val_manifest or (args.val_input_dir and args.val_target_dir):
        val_loader = DataLoader(
            FiveKPairedDataset(
                args.val_input_dir or args.input_dir,
                args.val_target_dir or args.target_dir,
                image_size=args.image_size,
                manifest=args.val_manifest,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    model = CLUTNet(args.architecture).to(device)
    if args.init_checkpoint:
        load_state_dict(
            model,
            args.init_checkpoint,
            device,
            allow_legacy_pickle=args.allow_legacy_checkpoint,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    perceptual_model = lpips.LPIPS(net="alex").to(device).eval()
    perceptual_model.requires_grad_(False)

    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        samples = 0
        for batch_index, batch in enumerate(train_loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs, inputs)["fakes"]
            l1_loss = (outputs - targets).abs().mean()
            perceptual = perceptual_model(outputs * 2 - 1, targets * 2 - 1).mean()
            loss = l1_loss + args.lambda_perc * perceptual
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(inputs)
            samples += len(inputs)

        train_loss = running / max(samples, 1)
        validation_loss = (
            evaluate(
                model,
                val_loader,
                perceptual_model,
                device,
                lambda_perc=args.lambda_perc,
                max_batches=args.max_batches,
            )
            if val_loader is not None
            else train_loss
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss}
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            model.load_state_dict(best_state)
            save_checkpoint(
                model,
                output_dir / "generic_enhancer_best.pt",
                epoch=epoch,
                architecture=args.architecture,
            )
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train={train_loss:.6f} | validation={validation_loss:.6f}"
        )

    model.load_state_dict(best_state)
    save_checkpoint(
        model,
        output_dir / "generic_enhancer_final.pt",
        history=history,
        architecture=args.architecture,
    )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--val-input-dir", type=Path)
    parser.add_argument("--val-target-dir", type=Path)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/generic_enhancer"))
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument(
        "--allow-legacy-checkpoint",
        action="store_true",
        help="Allow a trusted checkpoint that serializes a full Python model.",
    )
    parser.add_argument("--architecture", default="20+05+20")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lambda-perc", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--max-batches", type=int)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
