from __future__ import annotations

from PIL import Image

from datasets.fivek import FiveKPairedDataset
from datasets.split_fivek import split_fivek


def test_fivek_loader_matches_by_filename(tmp_path) -> None:
    input_dir = tmp_path / "input"
    target_dir = tmp_path / "target"
    input_dir.mkdir()
    target_dir.mkdir()
    Image.new("RGB", (16, 12), (10, 20, 30)).save(input_dir / "sample.png")
    Image.new("RGB", (16, 12), (15, 25, 35)).save(target_dir / "sample.png")

    dataset = FiveKPairedDataset(input_dir, target_dir, image_size=8)
    sample = dataset[0]
    assert len(dataset) == 1
    assert sample["name"] == "sample.png"
    assert sample["input"].shape == (3, 8, 8)
    assert sample["target"].shape == (3, 8, 8)


def test_fivek_manifest_split(tmp_path) -> None:
    input_dir = tmp_path / "input"
    target_dir = tmp_path / "target"
    split_dir = tmp_path / "splits"
    input_dir.mkdir()
    target_dir.mkdir()
    for index in range(4):
        name = f"image_{index}.png"
        Image.new("RGB", (8, 8), (index, index, index)).save(input_dir / name)
        Image.new("RGB", (8, 8), (index, index, index)).save(target_dir / name)

    summary = split_fivek(
        input_dir,
        target_dir,
        split_dir,
        train_ratio=0.5,
        val_ratio=0.25,
        seed=7,
    )
    assert summary["train_images"] == 2
    assert summary["val_images"] == 1
    assert summary["test_images"] == 1
    test_dataset = FiveKPairedDataset(
        input_dir,
        target_dir,
        manifest=split_dir / "test.txt",
    )
    assert len(test_dataset) == 1
