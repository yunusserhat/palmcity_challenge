"""Public-validation sweeps verified with local constant checkpoints and PNGs."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from palmcity.data import CLASS_NAMES, data_input_sha256
from palmcity.models import PREPROCESSING, TinySegmentationModel
from palmcity.training_state import file_sha256
from palmcity.validation_sweep import _validated_plan, default_plan


@pytest.fixture
def sweep_fixture(tmp_path, monkeypatch):
    sweep = importlib.import_module("palmcity.validation_sweep")
    prediction = importlib.import_module("palmcity.predict")
    storage = importlib.import_module("palmcity.storage")
    for module in (sweep, prediction, storage):
        monkeypatch.setattr(module, "require_workspace", lambda: tmp_path)
    image, mask = tmp_path / "sample.png", tmp_path / "sample-mask.png"
    Image.new("RGB", (16, 8), (128, 64, 0)).save(image)
    Image.fromarray(np.full((8, 16), 31, dtype=np.uint8)).save(mask)
    record = {"id": "sample", "image": str(image), "mask": str(mask)}
    manifest = {"train": [record], "val": [record], "test": []}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    audit = {"valid": True, "class_names": CLASS_NAMES, "counts": {"train": 1, "val": 1, "test": 0},
             "class_histograms": {"train": [0] * 31 + [128]},
             "manifest_sha256": file_sha256(manifest_path), "input_sha256": data_input_sha256(manifest),
             "validation_input_sha256": data_input_sha256(manifest, splits=("val",))}
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit))
    checkpoints = []
    for class_id in (0, 31):
        model = TinySegmentationModel()
        with torch.no_grad():
            model.layers[2].weight.zero_()
            model.layers[2].bias.fill_(-20)
            model.layers[2].bias[class_id] = 20
        config = {"image_size": [8, 16], "seed": 17,
                  "model": {"architecture": "tiny", "classes": 32, "encoder_weights": None},
                  "class_names": CLASS_NAMES, "preprocessing": PREPROCESSING}
        path = tmp_path / f"constant-{class_id}.pt"
        torch.save({"format_version": 1, "state_dict": model.state_dict(), "config": config,
                    "class_names": CLASS_NAMES, "preprocessing": PREPROCESSING}, path)
        checkpoints.append(str(path))
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield sweep, tmp_path, manifest_path, audit_path, checkpoints
    torch.set_num_threads(previous)


def test_validation_sweep_failure_resume_selection_and_png_provenance(sweep_fixture, monkeypatch):
    sweep, root, manifest, audit, checkpoints = sweep_fixture
    plan = {"variants": [
        {"name": "road", "checkpoints": [checkpoints[0]]},
        {"name": "void", "checkpoints": [checkpoints[1]]},
        {"name": "transient", "checkpoints": checkpoints, "weights": [0.25, 0.75]},
        {"name": "selected-flip", "select_best_of": ["road", "void"], "hflip_tta": True},
    ]}
    plan_path = root / "plan.json"
    plan_path.write_text(json.dumps(plan))
    original = sweep.predict
    calls = []

    def fail_once(*args, **kwargs):
        calls.append(str(args[2]))
        if "transient" in Path(args[2]).parts and "attempt-01" in Path(args[2]).parts:
            raise RuntimeError("A transient synthetic inference failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(sweep, "predict", fail_once)
    state_path = sweep.run_sweep(plan_path, manifest, audit, "sweep", device_name="cpu")
    state = json.loads(state_path.read_text())
    assert state["completed_variants"] == 3 and state["status"] == "partial"
    assert state["variants"]["transient"]["status"] == "failed"
    assert state["variants"]["void"]["miou"] == 1 / 32
    selected = state["variants"]["selected-flip"]["attempts"][-1]
    assert selected["resolved_variant"]["selected_source"] == "void"
    assert selected["resolved_variant"]["checkpoints"] == [checkpoints[1]]
    report = json.loads((root / "sweep" / "comparison.json").read_text())
    assert len(report["experiments"]) == 3
    assert report["experiments"][0]["class_iou_percent"][31] == 100
    first_calls = len(calls)
    sweep.run_sweep(plan_path, manifest, audit, "sweep", device_name="cpu", resume=True, retry_failed=True)
    resumed = json.loads(state_path.read_text())
    assert len(calls) == first_calls + 1  # completed variants were not re-inferred
    assert resumed["completed_variants"] == 4 and resumed["status"] == "complete"
    assert len(resumed["variants"]["transient"]["attempts"]) == 2
    assert (root / "sweep" / "transient" / "attempt-01").is_dir()
    assert (root / "sweep" / "transient" / "attempt-02" / "png" / "sample.png").is_file()
    (root / "sweep" / "comparison.json").unlink()
    sweep.run_sweep(plan_path, manifest, audit, "sweep", device_name="cpu", resume=True)
    assert len(calls) == first_calls + 1
    assert (root / "sweep" / "comparison.json").is_file()
    with pytest.raises(FileExistsError):
        sweep.run_sweep(plan_path, manifest, audit, "sweep", device_name="cpu")
    Image.fromarray(np.zeros((8, 16), dtype=np.uint8)).save(Path(selected["prediction_dir"]) / "sample.png")
    with pytest.raises(ValueError, match="PNGs changed"):
        sweep.run_sweep(plan_path, manifest, audit, "sweep", device_name="cpu", resume=True)


def test_validation_sweep_budget_defers_without_inference(sweep_fixture, monkeypatch):
    sweep, root, manifest, audit, checkpoints = sweep_fixture
    plan_path = root / "bounded.json"
    plan_path.write_text(json.dumps([{"name": "too-long", "checkpoints": [checkpoints[0]], "estimated_seconds": 100}]))
    monkeypatch.setattr(sweep, "predict", lambda *args, **kwargs: pytest.fail("Deferred variant ran inference"))
    path = sweep.run_sweep(plan_path, manifest, audit, "bounded", device_name="cpu", budget_seconds=1)
    state = json.loads(path.read_text())
    assert state["variants"]["too-long"]["status"] == "budget_deferred"
    assert state["variants"]["too-long"]["attempts"] == []
    assert not state["quality_metrics_measured"] and state["selected_variant"] is None


def test_prediction_time_limit_preserves_partial_outputs(sweep_fixture):
    _, root, manifest, _, checkpoints = sweep_fixture
    from palmcity.predict import predict

    with pytest.raises(ValueError, match="finite and positive"):
        predict([checkpoints[0]], manifest, "invalid-limit", split="val", device_name="cpu", time_limit_seconds=0)
    assert not (root / "invalid-limit").exists()
    with pytest.raises(TimeoutError, match="partial PNGs were preserved"):
        predict([checkpoints[0]], manifest, "deadline", split="val", device_name="cpu", time_limit_seconds=1e-12)
    assert (root / "deadline").is_dir()
    assert not (root / "deadline.json").exists()


@pytest.mark.parametrize("variants", [[], [{"name": f"v{i}", "checkpoints": ["local.pt"]} for i in range(13)],
                                     [{"name": "escape/name", "checkpoints": ["local.pt"]}],
                                     [{"name": "early", "select_best_of": ["not-yet-declared"]}],
                                     [{"name": "bad-weight", "checkpoints": ["local.pt"], "weights": [-1]}]])
def test_invalid_plans_rejected_before_processing(variants, tmp_path):
    with pytest.raises(ValueError):
        _validated_plan(variants, tmp_path)


def test_default_plan_is_fixed_and_probability_ensembles_are_declared(tmp_path):
    plan = default_plan("winner.pt", "runner.pt", ["seed0.pt", "seed1.pt", "seed2.pt"])
    variants = _validated_plan(plan, tmp_path)
    assert len(variants) == 11
    assert [v["weights"] for v in variants if v["name"].startswith("cross-")] == [[0.5, 0.5], [0.75, 0.25], [0.25, 0.75]]
    assert next(v for v in variants if v["name"] == "winner-window")["window_size"] == [256, 512]
    assert next(v for v in variants if v["name"] == "selected-ensemble-multiscale-flip")["scales"] == [0.75, 1.0, 1.25]
