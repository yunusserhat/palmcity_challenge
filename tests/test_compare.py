import json
from pathlib import Path

import numpy as np
import pytest

from palmcity.compare import comparison
from palmcity.data import CLASS_NAMES
from palmcity.metrics import scores_from_confusion


def fixtures(tmp_path):
    matrix = np.zeros((32, 32), dtype=np.int64)
    matrix[31, 31] = 20
    metrics = scores_from_confusion(matrix)
    score = {"split": "val", "manifest_sha256": "fixed-manifest", "images": 84,
             "prediction_set_sha256": "fixed-png-set",
             "validation_input_sha256": "fixed-val-input",
             "miou": metrics["miou"], "mf1": metrics["mf1"], "confusion_matrix": matrix.tolist(),
             "classes": [{"id": i, "name": name, "iou": metrics["iou"][i]} for i, name in enumerate(CLASS_NAMES)]}
    inference = {"split": "val", "manifest_sha256": "fixed-manifest", "num_predictions": 84,
                 "prediction_set_sha256": "fixed-png-set",
                 "split_input_sha256": "fixed-val-input",
                 "class_names": CLASS_NAMES}
    score_path, inference_path = tmp_path / "score.json", tmp_path / "inference.json"
    score_path.write_text(json.dumps(score))
    inference_path.write_text(json.dumps(inference))
    audit = {"valid": True, "class_names": CLASS_NAMES, "counts": {"val": 84},
             "manifest_sha256": "fixed-manifest",
             "validation_input_sha256": "fixed-val-input", "input_sha256": "fixed-all-input",
             "class_histograms": {"train": list(range(32))}}
    entry = {"name": "single", "score_report": str(score_path), "prediction_report": str(inference_path)}
    return entry, audit, score, score_path


def test_comparison_retains_void_and_train_defined_rare_classes(tmp_path):
    entry, audit, _, _ = fixtures(tmp_path)
    result = comparison([entry], audit)
    assert result["rare_class_ids"] == list(range(1, 9))
    assert result["experiments"][0]["miou_percent"] == 100 / 32
    assert result["experiments"][0]["class_iou_percent"][31] == 100


@pytest.mark.parametrize("field,value,match", [
    ("miou", 1.0, "macro metric"), ("split", "test", "validation"),
    ("manifest_sha256", "other", "same manifest"),
    ("prediction_set_sha256", "other", "same prediction PNG"),
    ("validation_input_sha256", "other", "same validation input"),
])
def test_comparison_rejects_invalid_evidence(tmp_path, field, value, match):
    entry, audit, score, path = fixtures(tmp_path)
    score[field] = value
    path.write_text(json.dumps(score))
    with pytest.raises(ValueError, match=match):
        comparison([entry], audit)


def test_training_metadata_must_match_actual_inference_config(tmp_path):
    entry, audit, _, _ = fixtures(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    metadata = {"seed": 42, "config_sha256": "config-a", "model_initialization": "random",
                "data": {"manifest_sha256": "fixed-manifest", "input_sha256": "fixed-all-input"}}
    (run / "metadata.json").write_text(json.dumps(metadata))
    (run / "summary.json").write_text("{}")
    entry["run_dirs"] = [str(run)]
    inference_path = Path(entry["prediction_report"])
    inference = json.loads(inference_path.read_text())
    inference["checkpoint_provenance"] = [{"path": str(run / "best.pt"), "config_sha256": "config-a"}]
    inference_path.write_text(json.dumps(inference))
    assert comparison([entry], audit)["experiments"][0]["training_runs"][0]["seed"] == 42
    metadata["config_sha256"] = "config-b"
    (run / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="does not match"):
        comparison([entry], audit)
