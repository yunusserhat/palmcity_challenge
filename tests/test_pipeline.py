"""A real CPU optimizer/inference round trip using synthetic images only."""

from __future__ import annotations

import importlib
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from palmcity.models import PREPROCESSING, RGB_MEAN, RGB_STD, image_tensor


def test_true_rgb_normalization_and_aspect_ratio() -> None:
    image = Image.new("RGB", (16, 8), (255, 0, 0))
    result = image_tensor(image, (8, 16))
    expected = (np.array([1.0, 0.0, 0.0]) - RGB_MEAN) / RGB_STD
    np.testing.assert_allclose(result[:, 0, 0].numpy(), expected, rtol=1e-6)
    with pytest.raises(ValueError, match="aspect ratio"):
        image_tensor(Image.new("RGB", (8, 8)), (8, 16))


def test_synthetic_train_predict_ensemble_and_no_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    training = importlib.import_module("palmcity.train")
    prediction = importlib.import_module("palmcity.predict")
    storage = importlib.import_module("palmcity.storage")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(training, "require_workspace", lambda: workspace)
    monkeypatch.setattr(prediction, "require_workspace", lambda: workspace)
    monkeypatch.setattr(storage, "require_workspace", lambda: workspace)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    records = []
    for number in range(3):
        image_path, mask_path = inputs / f"sample{number}.png", inputs / f"sample{number}_mask.png"
        image = np.zeros((8, 16, 3), dtype=np.uint8)
        image[:, :8, 0] = 255
        image[:, 8:, 1] = 255
        mask = np.zeros((8, 16), dtype=np.uint8)
        mask[:, 8:] = 31
        Image.fromarray(image).save(image_path)
        Image.fromarray(mask).save(mask_path)
        records.append({"id": f"sample{number}", "image": str(image_path), "mask": str(mask_path)})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"train": records[:2], "val": records[2:],
                                    "test": [{"id": "sample2", "image": records[2]["image"]}]}))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"image_size": [8, 16], "epochs": 1,
                                  "model": {"architecture": "tiny", "classes": 32, "encoder_weights": None},
                                  "batch_size": 1, "num_workers": 0, "amp": False,
                                  "augmentation": {"horizontal_flip": 0.0, "horizontal_roll": 0.0}}))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        checkpoint = training.train(config, manifest, "synthetic", device_name="cpu", smoke=True)
        assert checkpoint.is_file()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert saved["preprocessing"] == PREPROCESSING
        assert saved["config"]["model"]["architecture"] == "tiny"
        assert len(saved["class_names"]) == 32
        assert len((checkpoint.parent / "metrics.jsonl").read_text().splitlines()) == 1
        opposing_checkpoints = []
        for class_id in (0, 31):
            constant = copy.deepcopy(saved)
            constant["state_dict"]["layers.2.weight"].zero_()
            constant["state_dict"]["layers.2.bias"].fill_(-20)
            constant["state_dict"]["layers.2.bias"][class_id] = 20
            constant_path = workspace / f"constant-{class_id}.pt"
            torch.save(constant, constant_path)
            opposing_checkpoints.append(constant_path)
        output = prediction.predict(opposing_checkpoints, manifest, "predictions/synthetic",
                                    device_name="cpu", weights=[0.25, 0.75], hflip_tta=True, amp=False)
        with Image.open(output / "sample2.png") as image:
            assert image.mode == "L"
            assert image.size == (16, 8)
            # Probability weighting must let the stronger Void checkpoint win everywhere.
            assert np.all(np.asarray(image) == 31)
        with pytest.raises(FileExistsError):
            training.train(config, manifest, "synthetic", device_name="cpu", smoke=True)
        with pytest.raises(FileExistsError):
            prediction.predict([checkpoint], manifest, "predictions/synthetic", device_name="cpu", amp=False)
        saved["class_names"] = list(reversed(saved["class_names"]))
        invalid_checkpoint = workspace / "invalid.pt"
        torch.save(saved, invalid_checkpoint)
        with pytest.raises(ValueError, match="class order"):
            prediction.load_checkpoint(invalid_checkpoint)
        with pytest.raises(RuntimeError, match="Outputs must be inside"):
            prediction.predict([checkpoint], manifest, tmp_path / "outside", device_name="cpu", amp=False)
    finally:
        torch.set_num_threads(previous_threads)
