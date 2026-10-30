"""Joint crop geometry, train-only color changes, and resumable sampling."""

from __future__ import annotations

import importlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from palmcity.models import RGB_MEAN, RGB_STD, image_tensor
from palmcity.train import PanoramaDataset, build_training_loader


def panorama(tmp_path: Path, name: str = "sample") -> dict[str, str]:
    mask = np.tile(np.arange(32, dtype=np.uint8).reshape(4, 8), (2, 2))
    image = np.repeat(mask[..., None], 3, axis=-1)
    image_path, mask_path = tmp_path / f"{name}.png", tmp_path / f"{name}_mask.png"
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)
    return {"id": name, "image": str(image_path), "mask": str(mask_path)}


def augmented_config() -> dict:
    return {
        "image_size": [4, 8], "batch_size": 1, "num_workers": 0,
        "augmentation": {
            "horizontal_flip": 1.0, "horizontal_roll": 1.0,
            "scale_crop_probability": 1.0, "scale_crop_area": [0.25, 0.25],
            "color_jitter_probability": 0.0,
        },
    }


def assert_same_checkpoint_value(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same_checkpoint_value(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            assert_same_checkpoint_value(first, second)
    else:
        assert left == right


def test_crop_flip_and_roll_keep_rgb_and_all_32_id_masks_aligned(tmp_path: Path) -> None:
    dataset = PanoramaDataset([panorama(tmp_path)], augmented_config(), training=True)
    random.seed(713)
    image, target = dataset[0]
    # The crop is exactly 4x8, so neither RGB nor mask needs spatial resampling.
    restored_ids = ((image[0] * RGB_STD[0] + RGB_MEAN[0]) * 255).round().long()
    assert torch.equal(restored_ids, target)
    assert image.shape == (3, 4, 8)
    assert sorted(target.unique().tolist()) == list(range(32))


def test_validation_retains_full_native_mask_and_ignores_all_augmentations(tmp_path: Path) -> None:
    record, config = panorama(tmp_path), augmented_config()
    config["augmentation"].update({"color_jitter_probability": 1.0,
                                   "color_jitter_strength": [0.5, 0.5, 0.5, 0.1]})
    dataset = PanoramaDataset([record], config, training=False)
    state = random.getstate()
    image, target = dataset[0]
    assert random.getstate() == state
    assert target.shape == (8, 16)
    with Image.open(record["image"]) as original:
        assert torch.equal(image, image_tensor(original, (4, 8)))
    with Image.open(record["mask"]) as original:
        np.testing.assert_array_equal(target.numpy(), np.asarray(original))


def test_photometric_augmentation_changes_rgb_only_and_is_seeded(tmp_path: Path) -> None:
    record, config = panorama(tmp_path), augmented_config()
    config["image_size"] = [8, 16]
    config["augmentation"].update({"horizontal_flip": 0.0, "horizontal_roll": 0.0,
                                   "scale_crop_probability": 0.0,
                                   "color_jitter_probability": 1.0,
                                   "color_jitter_strength": [0.5, 0.5, 0.5, 0.1]})
    dataset = PanoramaDataset([record], config, training=True)
    random.seed(88)
    first_image, first_target = dataset[0]
    random.seed(88)
    second_image, second_target = dataset[0]
    assert torch.equal(first_image, second_image)
    assert torch.equal(first_target, second_target)
    with Image.open(record["mask"]) as original:
        np.testing.assert_array_equal(first_target.numpy(), np.asarray(original))
    with Image.open(record["image"]) as original:
        assert not torch.equal(first_image, image_tensor(original, (8, 16)))
    assert torch.isfinite(first_image).all()


@pytest.mark.parametrize("setting,value", [
    ("scale_crop_probability", float("nan")), ("color_jitter_probability", 1.1),
    ("scale_crop_area", [0, 1]), ("scale_crop_area", [0.8, 0.5]),
    ("scale_crop_area", None), ("color_jitter_strength", [0.1, 0.1, 0.1]),
    ("color_jitter_strength", [0.1, 0.1, 0.1, 0.6]),
    ("color_jitter_strength", [0.1, float("inf"), 0.1, 0]),
])
@pytest.mark.parametrize("training", [True, False])
def test_malformed_augmentation_fails_for_both_splits(setting, value, training) -> None:
    config = augmented_config()
    config["augmentation"][setting] = value
    with pytest.raises(ValueError):
        PanoramaDataset([], config, training=training)


def test_weighted_loader_shares_checkpointed_rng_and_epoch_length(tmp_path: Path) -> None:
    records = [panorama(tmp_path, f"sample-{index}") for index in range(5)]
    config = augmented_config()
    config["training_sampling"] = {"min_pixels": 1}
    dataset = PanoramaDataset(records, config, training=True)
    generator = torch.Generator().manual_seed(52)
    loader, metadata = build_training_loader(dataset, config, torch.device("cpu"), generator)
    assert loader.sampler.generator is generator
    assert len(loader.sampler) == len(records)
    assert loader.sampler.replacement
    assert metadata["mode"] == "rare_class"
    _ = list(loader.sampler)
    checkpointed_state = generator.get_state()
    expected_next_epoch = list(loader.sampler)
    restored = torch.Generator()
    restored.set_state(checkpointed_state)
    resumed_loader, _ = build_training_loader(dataset, config, torch.device("cpu"), restored)
    assert list(resumed_loader.sampler) == expected_next_epoch
    with pytest.raises(ValueError, match="training dataset"):
        build_training_loader(PanoramaDataset(records, config, training=False), config,
                              torch.device("cpu"), restored)


@pytest.mark.parametrize("workers", [0, 2])
def test_augmented_weighted_training_resume_matches_uninterrupted(tmp_path: Path, monkeypatch, workers) -> None:
    training = importlib.import_module("palmcity.train")
    storage = importlib.import_module("palmcity.storage")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(training, "require_workspace", lambda: workspace)
    monkeypatch.setattr(storage, "require_workspace", lambda: workspace)
    records = [panorama(tmp_path, f"train-{index}") for index in range(4)]
    for index, record in enumerate(records[:3]):
        labels = np.zeros((8, 16), dtype=np.uint8)
        labels[:4] = 7
        labels[6:] = 31
        if index == 0:
            labels[:2] = 11  # A real rare class occurs in one training mask only.
        Image.fromarray(labels).save(record["mask"])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"train": records[:3], "val": records[3:], "test": []}))
    config = augmented_config()
    config.update({"epochs": 3, "seed": 78, "amp": False, "num_workers": workers,
                   "deterministic": True, "learning_rate": 0.001,
                   "training_sampling": {"min_pixels": 1},
                   "model": {"architecture": "tiny", "classes": 32}})
    config["augmentation"].update({"color_jitter_probability": 1.0,
                                   "color_jitter_strength": [0.2, 0.2, 0.2, 0.05]})
    _, sampling_metadata = build_training_loader(
        PanoramaDataset(records[:3], config, training=True), config, torch.device("cpu"),
        torch.Generator().manual_seed(config["seed"]),
    )
    details = sampling_metadata["details"]
    assert details["class_image_counts"][11] == 1
    assert details["class_image_counts"][7] == 3
    assert details["weights"] == pytest.approx([3 ** 0.5, 1.0, 1.0])
    assert details["weight_max"] > details["weight_min"]
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        training.train(config_path, manifest, "continuous", device_name="cpu")
        training.train(config_path, manifest, "resumed", device_name="cpu", stop_after_epochs=1)
        training.train(config_path, manifest, "resumed", device_name="cpu",
                       resume=workspace / "runs/resumed/latest.pt")
        full = torch.load(workspace / "runs/continuous/latest.pt", weights_only=True)
        resumed = torch.load(workspace / "runs/resumed/latest.pt", weights_only=True)
        for key in ("state_dict", "optimizer", "scheduler", "scaler", "rng", "best_miou",
                    "best_epoch", "epoch", "optimizer_steps"):
            assert_same_checkpoint_value(full[key], resumed[key])
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(False)
