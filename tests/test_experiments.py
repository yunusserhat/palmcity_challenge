"""Bounded real-loader calibration, fixed queues and all-class aggregation."""

from __future__ import annotations

import copy
import importlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from palmcity.experiments import budget_from_calibrations
from palmcity.training_state import data_identity, file_sha256


@pytest.fixture
def experiment_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    experiments = importlib.import_module("palmcity.experiments")
    trainer = importlib.import_module("palmcity.train")
    storage = importlib.import_module("palmcity.storage")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for module in (experiments, trainer, storage):
        monkeypatch.setattr(module, "require_workspace", lambda: workspace)
    monkeypatch.setattr(experiments, "code_identity", lambda: {"sha256": "test-code"})
    monkeypatch.setattr(experiments, "dependency_versions", lambda: {"torch": "test-version"})
    records = []
    rng = np.random.default_rng(71)
    for index in range(7):
        image_path, mask_path = tmp_path / f"sample-{index}.png", tmp_path / f"mask-{index}.png"
        Image.fromarray(rng.integers(0, 256, (8, 16, 3), dtype=np.uint8)).save(image_path)
        Image.fromarray(rng.choice(np.array([0, 7, 31], dtype=np.uint8), (8, 16))).save(mask_path)
        records.append({"id": image_path.stem, "image": str(image_path), "mask": str(mask_path)})
    manifest_data = {"train": records[:5], "val": records[5:6], "test": [{"id": records[6]["id"], "image": records[6]["image"]}]}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(manifest_data))
    config = tmp_path / "tiny.json"
    payload = {"model": {"architecture": "tiny", "classes": 32, "encoder_weights": None},
               "image_size": [8, 16], "batch_size": 2, "gradient_accumulation": 2,
               "drop_last": False, "num_workers": 0, "learning_rate": 1e-3, "amp": False,
               "seed": 42, "augmentation": {"horizontal_flip": 0.5, "horizontal_roll": 0.5}}
    config.write_text(json.dumps(payload))
    report = {"schema_version": 1, "kind": "real_loader_calibration", "status": "ok",
              "candidate_name": "tiny", "config_path": str(config), "config_sha256": file_sha256(config),
              "model_config": payload["model"], "initialization": "random",
              "manifest_path": str(manifest), "data": data_identity(manifest_data, manifest),
              "image_size": [8, 16], "batch_size": 2, "gradient_accumulation": 2,
              "amp_dtype": "torch.float16", "training_update_seconds_median": 1.0,
              "inference_seconds_per_image": 0.01, "estimated_updates_per_epoch": 2}
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield experiments, trainer, workspace, config, manifest, report
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(False)


def test_real_loader_calibration_is_bounded_and_discards_quality(experiment_fixture) -> None:
    experiments, _, workspace, config, manifest, _ = experiment_fixture
    report = experiments.calibrate(config, manifest, "reports/calibration.json", device_name="cpu",
                                   warmup=1, steps=2, val_images=1)
    assert report["status"] == "ok"
    assert report["kind"] == "real_loader_calibration"
    assert report["quality_metrics"] is None
    assert report["optimizer_updates_executed"] == 3
    assert len(report["training_update_seconds"]) == 2
    assert report["validation_probe_images"] == 1
    assert report["data"]["counts"] == {"train": 5, "val": 1, "test": 1}
    assert len(report["data"]["input_files"]) == 12
    assert not (workspace / "runs").exists()
    assert report["training_peak_allocated_bytes"] == 0
    assert report["inference_seconds_per_image"] > 0
    with pytest.raises(FileExistsError):
        experiments.calibrate(config, manifest, "reports/calibration.json", device_name="cpu")
    with pytest.raises(ValueError, match="1..3 warmups"):
        experiments.calibrate(config, manifest, "reports/invalid.json", device_name="cpu", steps=99)


def test_compute_budget_preserves_equal_validation_opportunities_and_cohorts(experiment_fixture) -> None:
    _, _, _, _, _, fast = experiment_fixture
    slow = copy.deepcopy(fast)
    slow.update({"candidate_name": "slow", "initialization": "pretrained", "training_update_seconds_median": 2.0})
    missing = copy.deepcopy(fast)
    missing.update({"candidate_name": "gated", "initialization": "pretrained", "status": "failed", "reason": "access denied"})
    plan = budget_from_calibrations([fast, slow, missing], seconds=60, evaluations=10)
    assert {row["cohort"] for row in plan["runs"]} == {"random", "pretrained"}
    assert len(plan["unavailable_candidates"]) == 1
    for row in plan["runs"]:
        recipe = row["recipe"]
        assert len(recipe["validation_epochs"]) == len(set(recipe["validation_epochs"])) == 10
        assert recipe["validation_epochs"][-1] == recipe["epochs"]
        assert recipe["epochs"] * 2 >= recipe["max_optimizer_steps"]
        assert 0 < recipe["warmup_steps"] < recipe["max_optimizer_steps"]
        assert 0 <= 60 - row["estimated_train_seconds"] - row["estimated_validation_seconds"] < row["calibration"]["training_update_seconds_median"]
    with pytest.raises(ValueError, match="Need at least"):
        budget_from_calibrations([fast, slow], seconds=20, evaluations=10)
    altered = copy.deepcopy(slow)
    altered["image_size"] = [16, 32]
    with pytest.raises(ValueError, match="share resolution"):
        budget_from_calibrations([fast, altered])
    confirmation = budget_from_calibrations([fast], seconds=60, phase="confirmation")
    assert [row["seed"] for row in confirmation["runs"]] == [42, 123, 2026]


def test_durable_queue_skips_completed_runs_and_collects_all_class_metrics(experiment_fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    experiments, trainer, workspace, _, _, report = experiment_fixture
    calibration = workspace / "calibration.json"
    calibration.write_text(json.dumps(report))
    plan_path = experiments.write_plan([calibration], "planned", seconds=5, evaluations=2)
    launches = []

    class InlineProcess:
        pid = os.getpid()

        def __init__(self, command, **kwargs):
            launches.append(command)
            def argument(name):
                return command[command.index(name) + 1]
            trainer.train(argument("--config"), argument("--manifest"), argument("--run-name"),
                          device_name=argument("--device"),
                          resume=argument("--resume") if "--resume" in command else None)

        def wait(self):
            return 0

    monkeypatch.setattr(experiments.subprocess, "Popen", InlineProcess)
    # The code identity helper in the trainer uses subprocess.run internally;
    # stub it before replacing Popen globally through the shared subprocess module.
    monkeypatch.setattr(trainer, "code_identity", lambda: {"sha256": "test-code"})
    first = experiments.run_plan(plan_path, device_name="cpu")
    assert first["status"] == "complete"
    assert len(launches) == 1
    second = experiments.run_plan(plan_path, device_name="cpu")
    assert second["runs"][0]["status"] == "already_complete"
    assert len(launches) == 1
    result = experiments.aggregate(plan_path)
    assert result["split"] == "val"
    assert len(result["class_names"]) == 32
    assert len(result["runs"][0]["class_iou"]) == 32
    assert result["candidate_groups"][0]["seeds"] == [42]
    assert result["candidate_groups"][0]["miou_population_std"] == 0.0
    execution = plan_path.parent / "execution"
    assert (execution / "state-cpu.json").is_file()
    assert len(list(execution.glob("*.log"))) == 1
    run = result["runs"][0]
    summary_path = Path(run["run_dir"]) / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["best_miou"] = 0.987
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="Best summary"):
        experiments.aggregate(plan_path)


def test_queue_records_failure_and_does_not_claim_a_winner(experiment_fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    experiments, _, workspace, _, _, report = experiment_fixture
    calibration = workspace / "calibration.json"
    calibration.write_text(json.dumps(report))
    plan_path = experiments.write_plan([calibration], "fail-planned", seconds=5, evaluations=2)

    class FailedProcess:
        pid = os.getpid()

        def __init__(self, *args, **kwargs):
            pass

        def wait(self):
            return 7

    monkeypatch.setattr(experiments.subprocess, "Popen", FailedProcess)
    result = experiments.run_plan(plan_path, device_name="cpu")
    assert result["status"] == "finished_with_failures"
    assert result["runs"][0]["return_code"] == 7
    collected = experiments.aggregate(plan_path)
    assert collected["runs"][0]["status"] == "not_completed"
    assert "miou_mean" not in collected["candidate_groups"][0]


def test_queue_resumes_the_committed_epoch_without_changing_budget(experiment_fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    experiments, trainer, workspace, _, _, report = experiment_fixture
    calibration = workspace / "resume-calibration.json"
    calibration.write_text(json.dumps(report))
    plan_path = experiments.write_plan([calibration], "resume-planned", seconds=5, evaluations=2)
    launches = []
    monkeypatch.setattr(trainer, "code_identity", lambda: {"sha256": "test-code"})

    class InterruptedProcess:
        pid = os.getpid()

        def __init__(self, command, **kwargs):
            launches.append(command)
            def argument(name):
                return command[command.index(name) + 1]
            first = len(launches) == 1
            trainer.train(argument("--config"), argument("--manifest"), argument("--run-name"),
                          device_name=argument("--device"), stop_after_epochs=1 if first else None,
                          resume=argument("--resume") if "--resume" in command else None)
            self.return_code = 1 if first else 0

        def wait(self):
            return self.return_code

    monkeypatch.setattr(experiments.subprocess, "Popen", InterruptedProcess)
    first = experiments.run_plan(plan_path, device_name="cpu")
    assert first["status"] == "finished_with_failures"
    assert first["runs"][0]["summary"]["status"] == "paused"
    second = experiments.run_plan(plan_path, device_name="cpu")
    assert second["status"] == "complete"
    assert "--resume" in launches[1]
    assert second["runs"][0]["summary"]["optimizer_steps"] == 4
    assert second["runs"][0]["summary"]["validation_evaluations"] == 2
