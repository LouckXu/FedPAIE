"""Evaluate a FedPAIE enhancer against matched MIT-Adobe FiveK targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

from datasets.fivek import FiveKPairedDataset
from utils.checkpoints import load_clut_model
from utils.reproducibility import resolve_device


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, float | int | str | None]:
    device = resolve_device(args.device)
    model = load_clut_model(
        args.checkpoint,
        device,
        architecture=args.architecture,
        allow_legacy_pickle=args.allow_legacy_checkpoint,
    ).eval()
    dataset = FiveKPairedDataset(
        args.input_dir,
        args.target_dir,
        image_size=args.image_size,
        manifest=args.manifest,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    perceptual_model = None
    if not args.skip_lpips:
        import lpips

        perceptual_model = lpips.LPIPS(net="alex").to(device).eval()

    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []
    processed = 0
    for batch in loader:
        if args.max_images is not None and processed >= args.max_images:
            break
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        if args.max_images is not None:
            remaining = args.max_images - processed
            inputs, targets = inputs[:remaining], targets[:remaining]
        outputs = model(inputs, inputs)["fakes"].clamp(0.0, 1.0)
        mse = (outputs - targets).square().flatten(1).mean(1).clamp_min(1e-12)
        psnr_values.extend((10.0 * torch.log10(1.0 / mse)).cpu().tolist())
        if perceptual_model is not None:
            distances = perceptual_model(outputs * 2 - 1, targets * 2 - 1)
            lpips_values.extend(distances.view(-1).cpu().tolist())
        for output, target in zip(outputs, targets):
            output_array = output.permute(1, 2, 0).cpu().numpy()
            target_array = target.permute(1, 2, 0).cpu().numpy()
            ssim_values.append(
                float(
                    structural_similarity(
                        output_array,
                        target_array,
                        channel_axis=2,
                        data_range=1.0,
                    )
                )
            )
        processed += len(inputs)

    result: dict[str, float | int | str | None] = {
        "name": args.name,
        "num_images": processed,
        "image_size": args.image_size,
        "psnr": float(np.mean(psnr_values)) if psnr_values else None,
        "ssim": float(np.mean(ssim_values)) if ssim_values else None,
        "lpips": float(np.mean(lpips_values)) if lpips_values else None,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--name", default="personalized_enhancer")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--architecture",
        help="Optional override for legacy state-dict checkpoints without metadata.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--allow-legacy-checkpoint",
        action="store_true",
        help="Allow a trusted checkpoint that serializes a full Python model.",
    )
    return parser


def main() -> None:
    result = evaluate(build_parser().parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
