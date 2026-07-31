"""MIT-Adobe FiveK paired data loader."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional


class FiveKPairedDataset(Dataset):
    """Load matched input and expert-retouched images by filename."""

    def __init__(
        self,
        input_dir: str | Path,
        target_dir: str | Path,
        *,
        image_size: int | None = 256,
        manifest: str | Path | Sequence[str] | None = None,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.image_size = image_size
        extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        inputs = {
            path.name: path
            for path in self.input_dir.iterdir()
            if path.suffix.lower() in extensions
        }
        targets = {
            path.name: path
            for path in self.target_dir.iterdir()
            if path.suffix.lower() in extensions
        }
        matched_names = sorted(inputs.keys() & targets.keys())
        if isinstance(manifest, str | Path):
            manifest_names = [
                line.strip()
                for line in Path(manifest).read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        elif manifest is not None:
            manifest_names = [str(name) for name in manifest]
        else:
            manifest_names = matched_names
        missing_names = sorted(set(manifest_names) - set(matched_names))
        if missing_names:
            preview = ", ".join(missing_names[:3])
            raise FileNotFoundError(f"Manifest contains unmatched FiveK files: {preview}")
        self.names = manifest_names
        if not self.names:
            raise FileNotFoundError("No matched FiveK input/target pairs were found.")
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        name = self.names[index]
        with Image.open(self.inputs[name]) as image_file:
            input_image = functional.to_tensor(image_file.convert("RGB"))
        with Image.open(self.targets[name]) as target_file:
            target_image = functional.to_tensor(target_file.convert("RGB"))

        if self.image_size is not None:
            size = [self.image_size, self.image_size]
            input_image = functional.resize(input_image, size, antialias=True)
            target_image = functional.resize(target_image, size, antialias=True)
        elif input_image.shape[-2:] != target_image.shape[-2:]:
            target_image = functional.resize(
                target_image,
                list(input_image.shape[-2:]),
                antialias=True,
            )

        return {"input": input_image, "target": target_image, "name": name}
