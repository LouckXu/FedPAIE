"""Convert MIT-Adobe FiveK ProPhoto RGB TIFF images to resized sRGB JPEGs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import colour
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

PROPHOTO = colour.RGB_COLOURSPACES["ProPhoto RGB"]
SRGB = colour.RGB_COLOURSPACES["sRGB"]


def prophoto_to_srgb(array: np.ndarray) -> np.ndarray:
    normalized = array.astype(np.float32) / np.iinfo(array.dtype).max
    xyz = colour.RGB_to_XYZ(
        normalized,
        PROPHOTO,
        chromatic_adaptation_transform="Bradford",
    )
    srgb = colour.XYZ_to_RGB(xyz, SRGB, chromatic_adaptation_transform="Bradford")
    return np.clip(srgb * 255.0, 0.0, 255.0).astype(np.uint8)


def preprocess_fivek(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    collection_label: str,
    long_edge: int = 720,
    short_edge: int = 480,
    jpeg_quality: int = 100,
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    collection_label = re.sub(r"[^a-zA-Z0-9_-]+", "_", collection_label).strip("_")
    if not collection_label:
        raise ValueError("collection_label must contain at least one letter or number.")
    output_dir.mkdir(parents=True, exist_ok=True)

    log_rows = [["source", "output", "status", "reason", "original_size", "output_size"]]
    seen_names: set[str] = set()
    paths = sorted(input_dir.glob("*.tif")) + sorted(input_dir.glob("*.tiff"))
    for source_path in tqdm(paths, desc=f"FiveK {collection_label}"):
        output_name = source_path.stem.split("-")[0] + ".jpg"
        output_path = output_dir / output_name
        try:
            if output_name in seen_names:
                log_rows.append([source_path.name, output_name, "skip", "duplicate", "", ""])
                continue
            seen_names.add(output_name)
            if output_path.exists():
                log_rows.append([source_path.name, output_name, "skip", "exists", "", ""])
                continue

            with Image.open(source_path) as image_file:
                original_size = image_file.size
                array = np.asarray(image_file)
                if array.ndim == 2:
                    array = np.repeat(array[..., None], 3, axis=2)
                elif array.ndim == 3:
                    array = array[..., :3]
                else:
                    raise ValueError(f"Unsupported image shape: {array.shape}")
                if array.dtype == np.uint16:
                    array = prophoto_to_srgb(array)
                elif array.dtype != np.uint8:
                    maximum = float(array.max())
                    array = (
                        np.zeros_like(array, dtype=np.uint8)
                        if maximum == 0
                        else np.clip(array / maximum * 255.0, 0, 255).astype(np.uint8)
                    )
                if array.std() < 1e-3:
                    raise ValueError("near-constant image")
                image = Image.fromarray(array, mode="RGB")
                target_size = (
                    (short_edge, long_edge)
                    if image.height > image.width
                    else (long_edge, short_edge)
                )
                image = image.resize(target_size, Image.Resampling.LANCZOS)
                image.save(output_path, format="JPEG", quality=jpeg_quality)
            log_rows.append(
                [source_path.name, output_name, "ok", "", str(original_size), str(target_size)]
            )
        except (OSError, ValueError, UnidentifiedImageError) as error:
            log_rows.append([source_path.name, output_name, "skip", str(error), "", ""])

    with (output_dir / f"preprocess_log_{collection_label}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        csv.writer(stream).writerows(log_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--collection-label",
        required=True,
        help="Log label such as 'input' or 'expert_c'; it does not filter files.",
    )
    parser.add_argument("--long-edge", type=int, default=720)
    parser.add_argument("--short-edge", type=int, default=480)
    parser.add_argument("--jpeg-quality", type=int, default=100)
    args = parser.parse_args()
    preprocess_fivek(**vars(args))


if __name__ == "__main__":
    main()
