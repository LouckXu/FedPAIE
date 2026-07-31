"""Portable Flickr-AES client datasets for scorer and enhancer training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils.image_pipeline import IMAGENET_MEAN, IMAGENET_STD


def build_image_transform(image_size: int = 224, normalize: bool = True):
    operations: list[object] = [
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
    if normalize:
        operations.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(operations)


def _load_tensor_mapping(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


class FlickrAESDataset(Dataset):
    """One client split with optional cached color and semantic features."""

    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path,
        *,
        feature_root: str | Path | None = None,
        image_size: int = 224,
        normalize_images: bool = True,
        load_precomputed_features: bool = True,
        load_images: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)
        self.feature_root = Path(feature_root) if feature_root is not None else None
        self.transform = build_image_transform(image_size, normalize_images)
        self.load_precomputed_features = load_precomputed_features
        self.load_images = load_images

        self.records = pd.read_csv(self.csv_path)
        required = {"image_name", "score"}
        missing = required - set(self.records.columns)
        if missing:
            raise ValueError(f"{self.csv_path} is missing columns: {sorted(missing)}")
        self.records = self.records.dropna(subset=["image_name", "score"]).reset_index(drop=True)

        self.feature_files: dict[str, Path] = {}
        if load_precomputed_features:
            if self.feature_root is None:
                raise ValueError("feature_root is required when precomputed features are enabled.")
            index_path = self.feature_root / "feature_index.csv"
            index = pd.read_csv(index_path)
            if not {"image_name", "feature_path"}.issubset(index.columns):
                raise ValueError(f"Invalid feature index: {index_path}")
            self.feature_files = {
                str(row.image_name): self.feature_root / Path(
                    str(row.feature_path).replace("\\", "/")
                ).name
                for row in index.itertuples(index=False)
            }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.records.iloc[index]
        image_name = str(row["image_name"]).strip()
        image_path = self.image_root / image_name

        score = (float(row["score"]) - 1.0) / 4.0
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Rating outside 1-5 for {image_name}: {row['score']}")

        sample: dict[str, object] = {
            "score": torch.tensor(score, dtype=torch.float32),
            "client_id": str(row.get("client_id", "")),
            "image_name": image_name,
        }
        if self.load_images:
            with Image.open(image_path) as image_file:
                sample["image"] = self.transform(image_file.convert("RGB"))

        if self.load_precomputed_features:
            feature_path = self.feature_files.get(image_name)
            if feature_path is None or not feature_path.exists():
                raise FileNotFoundError(f"Missing cached features for {image_name}")
            features = _load_tensor_mapping(feature_path)
            color = features.get("color_features", features.get("colour_features"))
            semantic = features.get("semantic_features")
            if color is None or semantic is None:
                raise KeyError(f"Invalid cached features: {feature_path}")
            sample["color_features"] = color.float()
            sample["semantic_features"] = semantic.float()
        return sample


@dataclass(frozen=True)
class ClientLoaders:
    """The four disjoint per-client partitions used by FedPAIE."""

    train_fl: DataLoader
    personalization: DataLoader
    val: DataLoader
    test: DataLoader

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train_fl": len(self.train_fl.dataset),
            "personalization": len(self.personalization.dataset),
            "val": len(self.val.dataset),
            "test": len(self.test.dataset),
        }


def _split_path(split_root: Path, client_id: str | int, split: str) -> Path:
    path = split_root / f"{client_id}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Client split not found: {path}")
    return path


def build_client_loaders(
    data_root: str | Path,
    client_id: str | int,
    *,
    batch_size: int = 16,
    num_workers: int = 0,
    image_size: int = 224,
    normalize_images: bool = True,
    load_precomputed_features: bool = True,
    load_images: bool = True,
) -> ClientLoaders:
    """Build the 70/10/10/10 client partitions from a portable data root."""
    data_root = Path(data_root)
    split_root = data_root / "split" / "flickr_dp4_split"
    image_root = data_root / "raw" / "flickr_images"
    feature_root = data_root / "precomputed" / "flickr_features"

    def make(split: str, shuffle: bool) -> DataLoader:
        dataset = FlickrAESDataset(
            _split_path(split_root, client_id, split),
            image_root,
            feature_root=feature_root,
            image_size=image_size,
            normalize_images=normalize_images,
            load_precomputed_features=load_precomputed_features,
            load_images=load_images,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return ClientLoaders(
        train_fl=make("train_fl", True),
        personalization=make("personalization", True),
        val=make("val", False),
        test=make("test", False),
    )


def list_client_ids(data_root: str | Path) -> list[str]:
    """List numeric client IDs available under the split directory."""
    split_root = Path(data_root) / "split" / "flickr_dp4_split"
    client_ids = [
        path.name.removesuffix("_train_fl.csv")
        for path in split_root.glob("*_train_fl.csv")
    ]
    return sorted(
        client_ids,
        key=lambda value: (
            not value.isdigit(),
            int(value) if value.isdigit() else value,
        ),
    )
