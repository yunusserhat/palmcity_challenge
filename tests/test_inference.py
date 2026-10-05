"""Download-free checks of spatial alignment, overlap and official class order."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from palmcity.data import CLASS_NAMES, data_input_sha256, prediction_set_sha256
from palmcity.inference import InferenceOptions, inference_plans, panorama_probabilities, probabilities
from palmcity.models import PREPROCESSING, TinySegmentationModel


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class PixelClassifier(nn.Module):
    input_multiple = 8

    def __init__(self) -> None:
        super().__init__()
        self.shapes = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.shapes.append(tuple(images.shape[-2:]))
        logits = images.new_full((images.shape[0], 32, *images.shape[-2:]), -15)
        logits[:, 0] = images[:, 0] * 4
        logits[:, 31] = -images[:, 0] * 4
        return logits


def infer(model: nn.Module, image: torch.Tensor, **kwargs) -> torch.Tensor:
    return panorama_probabilities(model, image, tuple(image.shape[-2:]), device=torch.device("cpu"),
                                  use_amp=False, amp_dtype=torch.float16, **kwargs)


def test_flip_reversal_and_stride_padding_preserve_void_spatial_alignment() -> None:
    image = torch.zeros(3, 7, 14)
    image[0, :, :7] = 2
    image[0, :, 7:] = -2
    model = PixelClassifier()
    original = infer(model, image)
    flipped = infer(model, image, hflip_tta=True)
    torch.testing.assert_close(flipped, original)
    assert model.shapes == [(8, 16)] * 3
    expected = torch.zeros(7, 14, dtype=torch.long)
    expected[:, 7:] = 31
    assert torch.equal(flipped.argmax(0), expected)
    torch.testing.assert_close(flipped.sum(0), torch.ones(7, 14))


@pytest.mark.parametrize("overlap", [0.0, 0.37, 0.5, 0.8])
def test_overlap_covers_last_rows_and_columns_and_normalizes_probabilities(overlap: float) -> None:
    generator = torch.Generator().manual_seed(73)
    image = torch.randn(3, 17, 34, generator=generator)
    reference = infer(PixelClassifier(), image)
    options = InferenceOptions(window_size=(8, 16), overlap=overlap)
    result = infer(PixelClassifier(), image, hflip_tta=True, options=options)
    torch.testing.assert_close(result, reference, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(result.sum(0), torch.ones(17, 34))
    plan = inference_plans((17, 34), options)[0]
    assert plan["y_starts"][-1] == 9
    assert plan["x_starts"][-1] == 18


def test_multiscale_averages_softmax_probabilities_preserves_aspect_ratio() -> None:
    class ScaleClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.shapes = []

        def forward(self, images):
            height, width = images.shape[-2:]
            self.shapes.append((height, width))
            logits = images.new_full((1, 32, height, width), -10)
            logits[:, 0] = 2 if height == 4 else -3
            logits[:, 31] = 0 if height == 4 else 8
            return logits

    image = torch.zeros(3, 8, 16)
    model = ScaleClassifier()
    result = infer(model, image, options=InferenceOptions(scales=(0.5, 1.0)))
    low = torch.full((32,), -10.0)
    low[0], low[31] = 2, 0
    high = torch.full((32,), -10.0)
    high[0], high[31] = -3, 8
    expected = ((low.softmax(0) + high.softmax(0)) / 2)[:, None, None].expand_as(result)
    torch.testing.assert_close(result, expected)
    assert model.shapes == [(4, 8), (8, 16)]
    assert torch.all(result.argmax(0) == 31)


def test_scales_use_native_pixels_instead_of_upscaling_downsampled_checkpoint_input() -> None:
    image = torch.zeros(3, 8, 16)
    image[0, :, ::2] = 2
    image[0, :, 1::2] = -2
    reference = infer(PixelClassifier(), image)
    result = infer(PixelClassifier(), image, base_size=(4, 8), options=InferenceOptions(scales=(2,)))
    torch.testing.assert_close(result, reference)
    assert torch.all(result.argmax(0)[:, ::2] == 0)
    assert torch.all(result.argmax(0)[:, 1::2] == 31)


@pytest.mark.parametrize("options, message", [
    ({"scales": ()}, "scales"), ({"scales": (float("nan"),)}, "scales"),
    ({"scales": (0.0,)}, "scales"), ({"scales": (1.0,) * 9}, "scales"),
    ({"window_size": (8, 8)}, "aspect ratio"), ({"overlap": 1.0}, "overlap"),
    ({"max_tiles": 0}, "positive integer"), ({"max_scaled_pixels": -1}, "positive integer"),
])
def test_invalid_inference_options_are_rejected(options: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        InferenceOptions(**options)


def test_resource_bounds_are_checked_before_forward() -> None:
    model = PixelClassifier()
    image = torch.zeros(3, 16, 32)
    with pytest.raises(ValueError, match="max_scaled_pixels"):
        infer(model, image, options=InferenceOptions(scales=(2,), max_scaled_pixels=512))
    with pytest.raises(ValueError, match="max_tiles"):
        infer(model, image, options=InferenceOptions(window_size=(2, 4), max_tiles=2))
    assert model.shapes == []


def test_flip_output_must_keep_all_32_classes() -> None:
    class BadFlip(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, images):
            self.calls += 1
            return images.new_zeros((1, 32 if self.calls == 1 else 31, *images.shape[-2:]))

    with pytest.raises(RuntimeError, match="all 32"):
        probabilities(BadFlip(), torch.zeros(1, 3, 8, 16), (8, 16), device=torch.device("cpu"),
                      use_amp=False, amp_dtype=torch.float16, hflip_tta=True)


def test_prediction_cli_roundtrip_provenance_probability_ensemble_and_class_order(tmp_path: Path, monkeypatch) -> None:
    prediction = importlib.import_module("palmcity.predict")
    storage = importlib.import_module("palmcity.storage")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(prediction, "require_workspace", lambda: workspace)
    monkeypatch.setattr(storage, "require_workspace", lambda: workspace)
    image_path = workspace / "sample.png"
    Image.new("RGB", (32, 16), (128, 50, 0)).save(image_path)
    manifest_path = workspace / "manifest.json"
    manifest_path.write_text(json.dumps({"train": [], "val": [],
                                         "test": [{"id": "sample", "image": str(image_path)}]}))
    config = {"model": {"architecture": "tiny", "encoder_weights": None, "classes": 32},
              "image_size": [16, 32], "seed": 17, "preprocessing": PREPROCESSING,
              "class_names": list(CLASS_NAMES)}
    checkpoints = []
    payloads = []
    for class_id in (0, 31):
        model = TinySegmentationModel()
        with torch.no_grad():
            model.layers[2].weight.zero_()
            model.layers[2].bias.fill_(-20)
            model.layers[2].bias[class_id] = 20
        payload = {"format_version": 1, "class_names": list(CLASS_NAMES), "preprocessing": PREPROCESSING,
                   "config": copy.deepcopy(config), "state_dict": model.state_dict()}
        path = workspace / f"class-{class_id}.pt"
        torch.save(payload, path)
        checkpoints.append(path)
        payloads.append(payload)
    prediction.main(["--checkpoint", *map(str, checkpoints), "--weights", "1", "3",
                     "--manifest", str(manifest_path), "--output-dir", "prediction",
                     "--device", "cpu", "--no-amp", "--hflip-tta", "--scales", "0.5", "1", "1.5",
                     "--window-size", "8", "16", "--overlap", "0.37"])
    output = workspace / "prediction"
    assert list(output.iterdir()) == [output / "sample.png"]
    with Image.open(output / "sample.png") as image:
        assert image.mode == "L" and image.size == (32, 16)
        assert np.all(np.asarray(image) == 31)
    metadata = json.loads((workspace / "prediction.json").read_text())
    assert metadata["weights"] == [0.25, 0.75]
    assert metadata["inference"]["scales"] == [0.5, 1, 1.5]
    assert metadata["class_names"][-1] == "Void"
    assert len(metadata["checkpoint_provenance"]) == 2
    assert all(len(item["sha256"]) == 64 and item["config"]["seed"] == 17 for item in metadata["checkpoint_provenance"])
    assert len(metadata["provenance"]["manifest_sha256"]) == 64
    assert metadata["prediction_set_sha256"] == prediction_set_sha256(output, [{"id": "sample"}])
    assert metadata["split_input_sha256"] == data_input_sha256(json.loads(manifest_path.read_text()), splits=("test",))
    assert metadata["runtime"]["prediction_seconds"] > 0
    assert metadata["runtime"]["total_seconds"] >= metadata["runtime"]["prediction_seconds"]
    assert metadata["memory"]["peak_allocated_bytes"] is None
    assert metadata["provenance"]["dependencies"]["torch"]
    payloads[0]["class_names"] = list(reversed(CLASS_NAMES))
    invalid = workspace / "invalid.pt"
    torch.save(payloads[0], invalid)
    with pytest.raises(ValueError, match="class order"):
        prediction.load_checkpoint(invalid)
    with pytest.raises(ValueError, match="max_scaled_pixels"):
        prediction.predict(checkpoints, manifest_path, "bad-bounds", device_name="cpu", scales=[100])
    assert not (workspace / "bad-bounds").exists()
    # Saved local state is sufficient even when training used pretrained sources;
    # reconstruction must clear every backend's initial download field.
    pretrained_payload = copy.deepcopy(payloads[1])
    for field in ("encoder_weights", "pretrained_model_name_or_path", "pretrained_backbone_name_or_path"):
        pretrained_payload["config"]["model"][field] = "forbidden-download-source"
    pretrained_path = workspace / "previously-pretrained.pt"
    torch.save(pretrained_payload, pretrained_path)
    original_constructor = prediction.build_model
    constructed = []

    def assert_download_free(config):
        constructed.append(config)
        assert config["encoder_weights"] is None
        assert config["pretrained_model_name_or_path"] is None
        assert config["pretrained_backbone_name_or_path"] is None
        return original_constructor(config)

    monkeypatch.setattr(prediction, "build_model", assert_download_free)
    prediction.load_checkpoint(pretrained_path)
    assert len(constructed) == 1


def test_synthetic_benchmark_has_measured_costs_and_no_dataset_score(tmp_path: Path, monkeypatch) -> None:
    benchmark = importlib.import_module("palmcity.inference_benchmark")
    storage = importlib.import_module("palmcity.storage")
    monkeypatch.setattr(benchmark, "require_workspace", lambda: tmp_path)
    monkeypatch.setattr(storage, "require_workspace", lambda: tmp_path)
    benchmark.main(["--report", "synthetic.json", "--device", "cpu", "--image-size", "8", "16", "--repeats", "1"])
    report = json.loads((tmp_path / "synthetic.json").read_text())
    assert len(report["scenarios"]) == 4
    assert report["shape"] == [8, 16]
    assert "measured PalmCity score" in report["scope"]
    assert all(row["mean_seconds"] > 0 and row["peak_allocated_gib"] is None for row in report["scenarios"])
    with pytest.raises(FileExistsError):
        benchmark.synthetic_benchmark("synthetic.json", image_size=(8, 16))


def test_synthetic_gpu_preflight_refuses_occupancy_before_cuda_initialization(tmp_path: Path, monkeypatch) -> None:
    benchmark = importlib.import_module("palmcity.inference_benchmark")
    storage = importlib.import_module("palmcity.storage")
    monkeypatch.setattr(benchmark, "require_workspace", lambda: tmp_path)
    monkeypatch.setattr(storage, "require_workspace", lambda: tmp_path)
    selected = []

    def occupied_gpu(index):
        selected.append(index)
        raise RuntimeError("Selected GPU has a compute process; defer the probe")

    def forbidden_initialization(name):
        pytest.fail("CUDA initialization happened before occupancy preflight")

    monkeypatch.setattr(benchmark, "idle_gpu", occupied_gpu)
    monkeypatch.setattr(benchmark, "select_device", forbidden_initialization)
    with pytest.raises(RuntimeError, match="compute process"):
        benchmark.synthetic_benchmark("blocked.json", device_name="cuda:1", image_size=(8, 16))
    assert selected == [1]
    assert not (tmp_path / "blocked.json").exists()
    with pytest.raises(ValueError, match="explicit physical cuda:N"):
        benchmark.synthetic_benchmark("implicit.json", device_name="cuda", image_size=(8, 16))
    assert selected == [1]
