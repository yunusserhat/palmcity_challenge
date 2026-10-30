"""Rarity uses training masks alone; RNG state determines replacement draws."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from palmcity.sampling import make_training_sampler, training_sample_weights


def record(tmp_path: Path, identifier: str, labels: np.ndarray) -> dict[str, str]:
    path = tmp_path / f"{identifier}.png"
    Image.fromarray(labels.astype(np.uint8)).save(path)
    # Sampling must not require reading the RGB image.
    return {"id": identifier, "mask": str(path), "image": str(tmp_path / "unopened.jpg")}


def test_rare_class_presence_caps_weights_and_keeps_all_class_ids(tmp_path):
    masks = [np.zeros((8, 8), dtype=np.uint8) for _ in range(4)]
    masks[0][:2] = 11  # 16 pixels, appearing in one of four images.
    masks[1][0, 0] = 12  # A single pixel is below the presence threshold.
    masks[2][:2] = 31  # Void exists but is excluded only from rarity calculation.
    records = [record(tmp_path, f"train-{i}", mask) for i, mask in enumerate(masks)]
    before = [Path(item["mask"]).read_bytes() for item in records]
    weights, metadata = training_sample_weights(records, {"max_weight": 1.5})
    torch.testing.assert_close(weights, torch.tensor([1.5, 1, 1, 1], dtype=torch.float64))
    assert weights.device.type == "cpu"
    assert metadata["class_image_counts"][11] == 1
    assert metadata["class_image_counts"][12] == 0
    assert metadata["class_image_counts"][31] == 1
    assert metadata["class_pixel_counts"][31] == 16
    assert len(metadata["class_weights"]) == 32
    assert metadata["class_weights"][31] == 1
    assert [Path(item["mask"]).read_bytes() for item in records] == before
    json.dumps(metadata, allow_nan=False)


def test_inverse_image_frequency_and_explicit_void_inclusion(tmp_path):
    records = [record(tmp_path, f"train-{i}", np.full((4, 4), 31 if i == 0 else 0))
               for i in range(4)]
    weights, _ = training_sample_weights(records, {"excluded_class_ids": [0]})
    torch.testing.assert_close(weights, torch.tensor([2, 1, 1, 1], dtype=torch.float64))
    uniform, _ = training_sample_weights(records, {"excluded_class_ids": [], "power": 0})
    torch.testing.assert_close(uniform, torch.ones(4, dtype=torch.float64))


def test_only_explicit_training_masks_are_opened(tmp_path, monkeypatch):
    import palmcity.sampling as sampling

    train = [record(tmp_path, "training-only", np.full((4, 4), 7))]
    # An invalid held-out mask would fail if the implementation discovered/read it.
    (tmp_path / "val.png").write_text("not a PNG")
    opened = []
    original = sampling.load_mask

    def tracked(path):
        opened.append(Path(path))
        return original(path)

    monkeypatch.setattr(sampling, "load_mask", tracked)
    first, metadata = sampling.training_sample_weights(train, {})
    assert opened == [Path(train[0]["mask"])]
    assert metadata["record_ids"] == ["training-only"]
    assert "val.png" not in json.dumps(metadata)
    (tmp_path / "val.png").write_text("changed held-out data")
    second, repeated = sampling.training_sample_weights(train, {})
    assert torch.equal(first, second) and metadata == repeated
    with pytest.raises(ValueError, match="nonempty sequence"):
        sampling.training_sample_weights({"train": train, "val": [], "test": []}, {})
    with pytest.raises(ValueError, match="only training records"):
        sampling.training_sample_weights([{**train[0], "split": "val"}], {})


def test_sampler_replays_generator_state_and_preserves_epoch_size():
    weights = torch.tensor([1.0, 2.0, 5.0], dtype=torch.float64)
    rng = torch.Generator().manual_seed(532)
    sampler = make_training_sampler(weights, generator=rng)
    initial = rng.get_state().clone()
    first = list(sampler)
    next_epoch = rng.get_state().clone()
    second = list(sampler)
    rng.set_state(initial)
    assert list(sampler) == first
    rng.set_state(next_epoch)
    assert list(sampler) == second
    assert len(first) == len(second) == len(sampler) == len(weights)
    assert sampler.replacement is True
    assert all(0 <= index < len(weights) for index in first + second)


@pytest.mark.parametrize("config", [
    {"power": float("nan")}, {"power": float("inf")}, {"power": -0.1},
    {"power": 1.1}, {"power": True}, {"max_weight": float("nan")},
    {"max_weight": 0.5}, {"min_pixels": 0}, {"min_pixels": 1.5},
    {"min_pixels": True}, {"excluded_class_ids": [32]},
    {"excluded_class_ids": [31, 31]}, {"excluded_class_ids": [True]},
    {"excluded_class_ids": "31"}, {"min_pixel": 16},
])
def test_invalid_sampling_config_rejected_before_mask_read(tmp_path, config):
    with pytest.raises(ValueError, match="Sampling"):
        training_sample_weights([{"id": "missing", "mask": str(tmp_path / "missing.png")}], config)


def test_missing_duplicate_and_invalid_training_masks_rejected(tmp_path):
    with pytest.raises(ValueError, match="nonempty sequence"):
        training_sample_weights([], {})
    with pytest.raises(ValueError, match="missing absolute mask file"):
        training_sample_weights([{"id": "missing", "mask": str(tmp_path / "missing.png")}], {})
    item = record(tmp_path, "example", np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="IDs must be unique"):
        training_sample_weights([item, item], {})
    invalid = record(tmp_path, "invalid", np.full((4, 4), 32, dtype=np.uint8))
    with pytest.raises(ValueError, match="invalid class IDs"):
        training_sample_weights([invalid], {})


@pytest.mark.parametrize("weights", [
    torch.tensor([]), torch.tensor([[1.0]]), torch.tensor([0.0]),
    torch.tensor([-1.0]), torch.tensor([float("nan")]), torch.tensor([float("inf")]),
    torch.tensor([1 + 1j]),
])
def test_invalid_sampler_weights_rejected(weights):
    with pytest.raises(ValueError, match="Sampling weights"):
        make_training_sampler(weights, generator=torch.Generator().manual_seed(1))
