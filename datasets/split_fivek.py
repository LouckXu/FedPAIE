"""Create deterministic FiveK train/validation/test manifests without copying images."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def matched_names(input_dir: Path, target_dir: Path) -> list[str]:
    inputs = {
        path.name
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    targets = {
        path.name
        for path in target_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    names = sorted(inputs & targets)
    if not names:
        raise FileNotFoundError("No matched FiveK input/target filenames were found.")
    return names


def split_fivek(
    input_dir: Path,
    target_dir: Path,
    output_dir: Path,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, int | float]:
    """Write filename-only manifests; the test split receives the remainder."""
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Require positive train/validation ratios whose sum is below 1.")
    names = matched_names(input_dir, target_dir)
    random.Random(seed).shuffle(names)
    train_end = int(len(names) * train_ratio)
    val_end = train_end + int(len(names) * val_ratio)
    partitions = {
        "train": names[:train_end],
        "val": names[train_end:val_end],
        "test": names[val_end:],
    }
    if any(not values for values in partitions.values()):
        raise ValueError("The requested ratios produce an empty FiveK partition.")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_names in partitions.items():
        (output_dir / f"{split_name}.txt").write_text(
            "\n".join(split_names) + "\n",
            encoding="utf-8",
        )
    summary: dict[str, int | float] = {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": round(1.0 - train_ratio - val_ratio, 10),
        **{f"{name}_images": len(values) for name, values in partitions.items()},
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = split_fivek(**vars(args))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
