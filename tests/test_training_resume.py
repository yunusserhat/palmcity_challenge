"""Resuming, accumulation and native losses verified with tiny synthetic data."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from palmcity.train import model_training_loss


@pytest.fixture
def synthetic_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    training = importlib.import_module("palmcity.train")
    storage = importlib.import_module("palmcity.storage")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(training, "require_workspace", lambda: workspace)
    monkeypatch.setattr(storage, "require_workspace", lambda: workspace)
    records = []
    generator = np.random.default_rng(812)
    for number in range(6):
        image_path, mask_path = tmp_path / f"image-{number}.png", tmp_path / f"mask-{number}.png"
        Image.fromarray(generator.integers(0, 256, (8, 16, 3), dtype=np.uint8)).save(image_path)
        Image.fromarray(generator.choice(np.array([0, 7, 31], dtype=np.uint8), (8, 16))).save(mask_path)
        records.append({"id": image_path.stem, "image": str(image_path), "mask": str(mask_path)})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"train": records[:5], "val": records[5:], "test": []}))
    config = tmp_path / "config.json"
    payload = {"image_size": [8, 16], "epochs": 3, "batch_size": 2, "gradient_accumulation": 2,
               "seed": 513, "num_workers": 0, "amp": False, "learning_rate": 1e-3,
               "minimum_learning_rate": 1e-5, "scheduler": "cosine", "warmup_steps": 1,
               "deterministic": True, "max_grad_norm": 1000.0,
               "model": {"architecture": "tiny", "classes": 32, "encoder_weights": None},
               "augmentation": {"horizontal_flip": 0.5, "horizontal_roll": 0.5}}
    config.write_text(json.dumps(payload))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield training, workspace, config, manifest, payload, records
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(False)


def assert_same_state(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same_state(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_same_state(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("workers", [0, 2])
def test_resume_exactly_matches_uninterrupted_training(synthetic_training, workers: int) -> None:
    training, workspace, config, manifest, payload, _ = synthetic_training
    payload["num_workers"] = workers
    config.write_text(json.dumps(payload))
    training.train(config, manifest, "continuous", device_name="cpu")
    training.train(config, manifest, "resumed", device_name="cpu", stop_after_epochs=1)
    run = workspace / "runs" / "resumed"
    assert json.loads((run / "summary.json").read_text())["status"] == "paused"
    # These derived files may lag latest.pt when a process fails during commit.
    (run / "metrics.jsonl").write_text("")
    (run / "best.pt").unlink()
    training.train(config, manifest, "resumed", device_name="cpu", resume=run / "latest.pt")
    continuous = torch.load(workspace / "runs" / "continuous" / "latest.pt", weights_only=True)
    resumed = torch.load(run / "latest.pt", weights_only=True)
    for key in ("state_dict", "optimizer", "scheduler", "scaler", "rng", "best_miou", "best_epoch", "optimizer_steps"):
        assert_same_state(continuous[key], resumed[key])
    assert resumed["optimizer_steps"] == 6
    assert resumed["scheduler"]["last_epoch"] == 6
    assert len((run / "metrics.jsonl").read_text().splitlines()) == 3
    assert set(path.name for path in run.glob("*.pt")) == {"best.pt", "latest.pt"}
    metadata = json.loads((run / "metadata.json").read_text())
    assert len(metadata["data"]["input_files"]) == 12
    assert metadata["data"]["manifest_sha256"]
    assert metadata["code"]["sha256"]
    assert metadata["dependencies"]["torch"]
    assert json.loads((run / "summary.json").read_text())["status"] == "complete"


def test_resume_rejects_changed_config_and_changed_data(synthetic_training) -> None:
    training, workspace, config, manifest, payload, records = synthetic_training
    training.train(config, manifest, "strict", device_name="cpu", stop_after_epochs=1)
    resume = workspace / "runs" / "strict" / "latest.pt"
    payload["learning_rate"] = 0.02
    config.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Resume identity differs"):
        training.train(config, manifest, "strict", device_name="cpu", resume=resume)
    payload["learning_rate"] = 1e-3
    config.write_text(json.dumps(payload))
    Image.new("RGB", (16, 8), (13, 0, 0)).save(records[0]["image"])
    with pytest.raises(ValueError, match="Resume identity differs"):
        training.train(config, manifest, "strict", device_name="cpu", resume=resume)
    with pytest.raises(ValueError, match="latest.pt in the same"):
        training.train(config, manifest, "strict", device_name="cpu", resume=resume.parent / "best.pt")


def test_accumulation_weights_short_batches_like_effective_batch(synthetic_training) -> None:
    training, workspace, config, manifest, payload, _ = synthetic_training
    payload.update({"epochs": 1, "warmup_steps": 0, "scheduler": "constant", "gradient_accumulation": 3,
                    "augmentation": {"horizontal_flip": 0.0, "horizontal_roll": 0.0}})
    config.write_text(json.dumps(payload))
    training.train(config, manifest, "microbatches", device_name="cpu")
    payload.update({"batch_size": 6, "gradient_accumulation": 1})
    config.write_text(json.dumps(payload))
    training.train(config, manifest, "effective_batch", device_name="cpu")
    micro = torch.load(workspace / "runs" / "microbatches" / "latest.pt", weights_only=True)
    full = torch.load(workspace / "runs" / "effective_batch" / "latest.pt", weights_only=True)
    assert micro["optimizer_steps"] == full["optimizer_steps"] == 1
    for key in micro["state_dict"]:
        torch.testing.assert_close(micro["state_dict"][key], full["state_dict"][key], rtol=1e-6, atol=1e-7)


def test_fixed_optimizer_budget_cuts_epoch_at_update_boundary(synthetic_training) -> None:
    training, workspace, config, manifest, payload, _ = synthetic_training
    payload.update({"batch_size": 1, "gradient_accumulation": 2, "max_optimizer_steps": 4,
                    "validation_every_epochs": 4})
    config.write_text(json.dumps(payload))
    training.train(config, manifest, "budget", device_name="cpu")
    latest = torch.load(workspace / "runs" / "budget" / "latest.pt", weights_only=True)
    assert latest["optimizer_steps"] == latest["scheduler"]["last_epoch"] == 4
    assert latest["metrics_history"][-1]["epoch_complete"] is False
    assert latest["metrics_history"][-1]["train_samples"] == 2
    assert latest["metrics_history"][0]["val_miou"] is None
    assert latest["metrics_history"][-1]["validation_performed"] is True
    before = latest["state_dict"]
    training.train(config, manifest, "budget", device_name="cpu", resume=workspace / "runs" / "budget" / "latest.pt")
    after = torch.load(workspace / "runs" / "budget" / "latest.pt", weights_only=True)["state_dict"]
    assert_same_state(before, after)


def test_reserved_validation_schedule_resumes_before_first_score(synthetic_training) -> None:
    training, workspace, config, manifest, payload, _ = synthetic_training
    payload["validation_epochs"] = [2, 3]
    config.write_text(json.dumps(payload))
    training.train(config, manifest, "reserved_continuous", device_name="cpu")
    best = training.train(config, manifest, "reserved_resume", device_name="cpu", stop_after_epochs=1)
    assert not best.exists()
    run = best.parent
    assert (run / "latest.pt").is_file()
    paused = json.loads((run / "summary.json").read_text())
    assert paused["best_miou"] is None
    assert paused["validation_evaluations"] == 0
    training.train(config, manifest, "reserved_resume", device_name="cpu", resume=run / "latest.pt")
    left = torch.load(workspace / "runs" / "reserved_continuous" / "latest.pt", weights_only=True)
    right = torch.load(run / "latest.pt", weights_only=True)
    for key in ("state_dict", "optimizer", "scheduler", "scaler", "rng", "best_miou", "best_epoch"):
        assert_same_state(left[key], right[key])
    assert json.loads((run / "summary.json").read_text())["validation_evaluations"] == 2
    assert [row["validation_performed"] for row in right["metrics_history"]] == [False, True, True]


def test_native_training_loss_dispatch_keeps_all_official_ids() -> None:
    class NativeModel(torch.nn.Module):
        uses_native_loss = True

        def training_loss(self, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            assert targets.max().item() == 31
            return images.square().mean()

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            raise AssertionError("Native loss must not use the inference conversion")

    images = torch.ones(1, 3, 2, 4, requires_grad=True)
    targets = torch.full((1, 2, 4), 31, dtype=torch.int64)
    loss = model_training_loss(NativeModel(), images, targets, 0.0)
    loss.backward()
    assert images.grad is not None
    with pytest.raises(ValueError, match="dice_weight=0"):
        model_training_loss(NativeModel(), images, targets, 0.5)


def test_query_progress_uses_optimizer_updates_not_microbatches(synthetic_training, monkeypatch: pytest.MonkeyPatch) -> None:
    training, _, config, manifest, payload, _ = synthetic_training
    progress: list[tuple[int, int]] = []

    class NativeModel(torch.nn.Module):
        uses_native_loss = True

        def __init__(self) -> None:
            super().__init__()
            self.head = torch.nn.Conv2d(3, 32, 1)

        def set_training_progress(self, completed: int, total: int) -> None:
            progress.append((completed, total))

        def training_loss(self, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.cross_entropy(self(images), targets)

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return self.head(images)

    monkeypatch.setattr(training, "build_model", lambda *args, **kwargs: NativeModel())
    payload.update({"epochs": 1, "warmup_steps": 0})
    config.write_text(json.dumps(payload))
    training.train(config, manifest, "progress", device_name="cpu")
    assert progress == [(0, 2), (0, 2), (1, 2)]
