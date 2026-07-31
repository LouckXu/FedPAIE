"""Precompute the fixed Flickr-AES scorer descriptors."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from utils.image_pipeline import (
    HSVLabFeatureExtractor,
    MobileNetV3FeatureExtractor,
    imagenet_normalize,
)
from utils.reproducibility import resolve_device


def _feature_name(image_name: str) -> str:
    return hashlib.sha256(image_name.encode("utf-8")).hexdigest()[:32] + ".pt"


@torch.no_grad()
def precompute_features(
    data_root: str | Path,
    *,
    image_size: int = 224,
    device: str = "auto",
    overwrite: bool = False,
) -> None:
    data_root = Path(data_root)
    split_root = data_root / "split" / "flickr_dp4_split"
    image_root = data_root / "raw" / "flickr_images"
    output_root = data_root / "precomputed" / "flickr_features"
    output_root.mkdir(parents=True, exist_ok=True)

    frames = []
    for csv_path in split_root.glob("*.csv"):
        split_names = ("train_fl", "personalization", "val", "test")
        if any(csv_path.name.endswith(f"_{name}.csv") for name in split_names):
            frame = pd.read_csv(csv_path, usecols=["image_name"])
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No client split CSV files were found in {split_root}")
    image_names = sorted(pd.concat(frames)["image_name"].dropna().astype(str).unique())

    runtime_device = resolve_device(device)
    color_extractor = HSVLabFeatureExtractor().to(runtime_device).eval()
    semantic_extractor = MobileNetV3FeatureExtractor().to(runtime_device).eval()
    semantic_transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(image_size), transforms.ToTensor()]
    )
    to_tensor = transforms.ToTensor()
    index_rows = []

    for image_name in tqdm(image_names, desc="Flickr-AES features"):
        output_path = output_root / _feature_name(image_name)
        if overwrite or not output_path.exists():
            with Image.open(image_root / image_name) as image_file:
                image = image_file.convert("RGB")
                color_input = to_tensor(image).to(runtime_device)
                semantic_input = semantic_transform(image).to(runtime_device)
            payload = {
                "image_name": image_name,
                "color_features": color_extractor(color_input).squeeze(0).cpu(),
                "semantic_features": semantic_extractor(
                    imagenet_normalize(semantic_input)
                ).squeeze(0).cpu(),
            }
            torch.save(payload, output_path)
        index_rows.append({"image_name": image_name, "feature_path": output_path.name})

    pd.DataFrame(index_rows).to_csv(output_root / "feature_index.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    precompute_features(**vars(args))


if __name__ == "__main__":
    main()
