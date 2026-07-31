"""Clean Flickr-AES labels and create deterministic 70/10/10/10 client splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

SPLIT_RATIOS = {
    "train_fl": 0.70,
    "personalization": 0.10,
    "val": 0.10,
    "test": 0.10,
}


def _split_sizes(sample_count: int) -> dict[str, int]:
    if sample_count < 4:
        raise ValueError("At least four ratings are required for four non-empty splits.")
    sizes = {
        "train_fl": round(sample_count * SPLIT_RATIOS["train_fl"]),
        "personalization": round(sample_count * SPLIT_RATIOS["personalization"]),
        "val": round(sample_count * SPLIT_RATIOS["val"]),
    }
    sizes["test"] = sample_count - sum(sizes.values())
    for name in sizes:
        sizes[name] = max(1, sizes[name])
    while sum(sizes.values()) > sample_count:
        for name in ("train_fl", "personalization", "val", "test"):
            if sizes[name] > 1 and sum(sizes.values()) > sample_count:
                sizes[name] -= 1
    while sum(sizes.values()) < sample_count:
        sizes["train_fl"] += 1
    return sizes


def _readable_rgb(path: Path) -> tuple[bool, int | None, int | None, str | None]:
    try:
        with Image.open(path) as image:
            image.convert("RGB")
            return True, image.width, image.height, image.mode
    except (OSError, ValueError, UnidentifiedImageError):
        return False, None, None, None


def preprocess_flickr_aes(
    data_root: str | Path,
    *,
    labels_csv: str | Path | None = None,
    image_root: str | Path | None = None,
    seed: int = 42,
    min_samples_per_client: int = 10,
    skip_image_check: bool = False,
    overwrite: bool = False,
) -> None:
    """Create numeric client splits without persisting original worker IDs."""
    data_root = Path(data_root)
    labels_csv = (
        Path(labels_csv)
        if labels_csv
        else data_root / "raw" / "FLICKR-AES_image_labeled_by_each_worker.csv"
    )
    image_root = Path(image_root) if image_root else data_root / "raw" / "flickr_images"
    processed_root = data_root / "processed" / "flickr"
    split_root = data_root / "split" / "flickr_dp4_split"

    existing_splits = list(split_root.glob("*_train_fl.csv")) if split_root.exists() else []
    if existing_splits and not overwrite:
        raise FileExistsError(
            f"Client splits already exist in {split_root}; pass --overwrite to replace them."
        )
    processed_root.mkdir(parents=True, exist_ok=True)
    split_root.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(labels_csv)
    frame.columns = frame.columns.str.strip()
    required = {"worker", "imagePair", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing Flickr-AES columns: {sorted(missing)}")
    frame = frame.rename(columns={"worker": "worker_id", "imagePair": "image_name"})
    frame = frame[["worker_id", "image_name", "score"]].copy()
    frame["worker_id"] = frame["worker_id"].astype(str).str.strip()
    frame["image_name"] = frame["image_name"].astype(str).str.strip()
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame = frame.dropna().drop_duplicates(subset=["worker_id", "image_name"], keep="first")
    frame = frame[frame["score"].between(1, 5)]

    users = sorted(frame["worker_id"].unique())
    frame["client_id"] = frame["worker_id"].map(
        {worker_id: index + 1 for index, worker_id in enumerate(users)}
    )

    if not skip_image_check:
        metadata: dict[str, tuple[bool, int | None, int | None, str | None]] = {}
        for image_name in tqdm(sorted(frame["image_name"].unique()), desc="Checking Flickr images"):
            metadata[image_name] = _readable_rgb(image_root / image_name)
        frame["readable"] = frame["image_name"].map(lambda name: metadata[name][0])
        frame["width"] = frame["image_name"].map(lambda name: metadata[name][1])
        frame["height"] = frame["image_name"].map(lambda name: metadata[name][2])
        frame["mode"] = frame["image_name"].map(lambda name: metadata[name][3])
        frame = frame[frame["readable"]].drop(columns=["readable"])

    # Raw worker identifiers are intentionally discarded before any output is written.
    frame = frame.drop(columns=["worker_id"]).reset_index(drop=True)
    frame.to_csv(processed_root / "records.csv", index=False)

    summaries = []
    skipped = []
    for client_id, client_frame in frame.groupby("client_id", sort=True):
        client_frame = client_frame.reset_index(drop=True)
        if len(client_frame) < min_samples_per_client:
            skipped.append({"client_id": client_id, "samples": len(client_frame)})
            continue
        sizes = _split_sizes(len(client_frame))
        shuffled = client_frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        offset = 0
        for split_name in ("train_fl", "personalization", "val", "test"):
            next_offset = offset + sizes[split_name]
            shuffled.iloc[offset:next_offset].to_csv(
                split_root / f"{client_id}_{split_name}.csv", index=False
            )
            offset = next_offset
        summaries.append({"client_id": client_id, "samples": len(client_frame), **sizes})

    pd.DataFrame(summaries).to_csv(split_root / "split_summary.csv", index=False)
    pd.DataFrame(skipped, columns=["client_id", "samples"]).to_csv(
        split_root / "skipped_clients.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--labels-csv", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-samples-per-client", type=int, default=10)
    parser.add_argument("--skip-image-check", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    preprocess_flickr_aes(**vars(args))


if __name__ == "__main__":
    main()
