"""Offline evidence that safe Hub conversion preserves trained panorama inference."""

from __future__ import annotations

import copy
import json
import socket
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors import safe_open

from palmcity.data import CLASS_NAMES
from palmcity.export_hub import export_checkpoint
from palmcity.hub import (input_images, load_released_model, read_release_metadata,
                         released_probabilities, resolve_release)
from palmcity.models import PREPROCESSING, build_model


@pytest.fixture
def synthetic_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def reject_network(*args, **kwargs):
        raise AssertionError("Hub release offline verification accessed the network")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setenv("PALMCITY_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("PALMCITY_WORF_PROFILE", "0")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    model_config = {"backend": "transformers", "architecture": "eomt_dinov3",
                    "variant": "custom", "classes": 32,
                    "hf_config": {"hidden_size": 32, "intermediate_size": 64,
                                  "num_attention_heads": 4, "num_hidden_layers": 2,
                                  "num_register_tokens": 4, "patch_size": 8,
                                  "image_size": 32, "num_queries": 4, "num_blocks": 1,
                                  "train_num_points": 16, "num_upscale_blocks": 1}}
    torch.manual_seed(53)
    model = build_model(model_config).eval()
    config = {"model": model_config, "seed": 2026, "image_size": [32, 64],
              "class_names": list(CLASS_NAMES), "preprocessing": PREPROCESSING,
              "pretrained_source": {"repo": "example/local-random-model",
                                    "revision": "0" * 40}}
    checkpoint = tmp_path / "synthetic.pt"
    torch.save({"format_version": 1, "config": config, "class_names": list(CLASS_NAMES),
                "preprocessing": PREPROCESSING, "state_dict": model.state_dict(),
                "optimizer": {"private_training_path": "/never/publish/this"}}, checkpoint)
    release = tmp_path / "seed2026"
    report = export_checkpoint(checkpoint, release)
    try:
        yield model, config, release, report
    finally:
        torch.set_num_threads(previous)


def test_safe_export_contains_native_hf_weights_without_training_state(synthetic_release) -> None:
    model, _, release, report = synthetic_release
    assert report["tensor_equality"]
    assert {path.name for path in release.iterdir()} == {
        "config.json", "model.safetensors", "palmcity_inference.json", "preprocessor_config.json"}
    assert "private_training_path" not in "".join(
        path.read_text() for path in release.glob("*.json"))
    with safe_open(release / "model.safetensors", framework="pt") as saved:
        for name, expected in model.network.state_dict().items():
            assert torch.equal(saved.get_tensor(name), expected)
    config = json.loads((release / "config.json").read_text())
    assert config["id2label"]["31"] == "Void"
    assert config["model_type"] == "eomt_dinov3"


def test_native_hf_roundtrip_preserves_rectangular_queries_and_multiscale(synthetic_release) -> None:
    model, config, release, _ = synthetic_release
    restored, metadata = load_released_model(release)
    image = Image.fromarray(np.random.default_rng(47).integers(0, 256, (35, 70, 3), dtype=np.uint8))
    expected = released_probabilities(
        [(model, config)], image, device=torch.device("cpu"),
        scales=(0.75, 1.0, 1.25), hflip_tta=True, amp=False)
    actual = released_probabilities(
        [(restored, metadata)], image, device=torch.device("cpu"),
        scales=(0.75, 1.0, 1.25), hflip_tta=True, amp=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual.argmax(0), expected.argmax(0))
    torch.testing.assert_close(actual.sum(0), torch.ones(35, 70), rtol=1e-5, atol=1e-5)
    assert restored.network.attn_mask_probs.eq(0).all()
    assert restored.network.grid_size[1] == 2 * restored.network.grid_size[0]


def test_release_probability_ensemble_averages_probabilities(synthetic_release) -> None:
    model, config, _, _ = synthetic_release
    altered = copy.deepcopy(model)
    with torch.no_grad():
        altered.network.class_predictor.bias[31] += 2
    image = Image.fromarray(np.zeros((32, 64, 3), dtype=np.uint8))
    options = {"device": torch.device("cpu"), "amp": False}
    first = released_probabilities([(model, config)], image, **options)
    second = released_probabilities([(altered, config)], image, **options)
    ensemble = released_probabilities([(model, config), (altered, config)], image, **options)
    torch.testing.assert_close(ensemble, (first + second) / 2, rtol=0, atol=0)


@pytest.mark.parametrize("field,value,message", [
    ("class_names", ["wrong"] * 32, "class order"),
    ("preprocessing", {}, "normalization"),
    ("image_size", [32, 32], "2:1"),
])
def test_incompatible_release_metadata_is_rejected(synthetic_release, field, value, message) -> None:
    _, _, release, _ = synthetic_release
    path = release / "palmcity_inference.json"
    metadata = json.loads(path.read_text())
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=message):
        read_release_metadata(release)


def test_local_release_needs_no_hub_request(synthetic_release) -> None:
    _, _, release, _ = synthetic_release
    assert resolve_release(release, (2026,)) == release.resolve()


def test_input_preflight_rejects_overwrites_and_wrong_panorama(tmp_path: Path) -> None:
    Image.new("RGB", (64, 32)).save(tmp_path / "same.png")
    Image.new("RGB", (64, 32)).save(tmp_path / "same.jpg")
    with pytest.raises(ValueError, match="unique"):
        input_images(tmp_path)
    wrong = tmp_path / "wrong.png"
    Image.new("RGB", (32, 32)).save(wrong)
    with pytest.raises(ValueError, match="2:1"):
        input_images(wrong)
