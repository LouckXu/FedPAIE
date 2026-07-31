"""Enhance one image or a directory with a personalized FedPAIE enhancer."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional

from utils.checkpoints import load_clut_model
from utils.reproducibility import resolve_device

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _image_paths(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path.suffix}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {path}")
    paths = sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError("No supported images were found in the input directory.")
    return paths


@torch.inference_mode()
def enhance(
    checkpoint: Path,
    input_path: Path,
    output_dir: Path,
    *,
    architecture: str | None,
    device: torch.device,
    max_images: int | None,
    allow_legacy_checkpoint: bool,
) -> list[Path]:
    """Run personalized enhancement while preserving each input resolution."""
    model = load_clut_model(
        checkpoint,
        device,
        architecture=architecture,
        allow_legacy_pickle=allow_legacy_checkpoint,
    ).eval()
    paths = _image_paths(input_path)
    if max_images is not None:
        paths = paths[:max_images]
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for path in paths:
        with Image.open(path) as image_file:
            image = ImageOps.exif_transpose(image_file).convert("RGB")
            tensor = functional.to_tensor(image).unsqueeze(0).to(device)
        enhanced = model(tensor, tensor)["fakes"].clamp(0.0, 1.0).squeeze(0).cpu()
        destination = output_dir / f"{path.stem}_enhanced.png"
        functional.to_pil_image(enhanced).save(destination)
        written.append(destination)
        print(f"Enhanced {path.name} -> {destination.name}")
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="An image or a flat image directory.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inference"))
    parser.add_argument(
        "--architecture",
        help="Optional override for legacy state-dict checkpoints without metadata.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-images", type=int)
    parser.add_argument(
        "--allow-legacy-checkpoint",
        action="store_true",
        help="Allow a trusted checkpoint that serializes a full Python model.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    enhance(
        args.checkpoint,
        args.input,
        args.output_dir,
        architecture=args.architecture,
        device=resolve_device(args.device),
        max_images=args.max_images,
        allow_legacy_checkpoint=args.allow_legacy_checkpoint,
    )


if __name__ == "__main__":
    main()
