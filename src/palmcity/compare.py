"""Compare a small, declared set of validation single/ensemble/TTA experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import CLASS_NAMES, NUM_CLASSES
from .metrics import scores_from_confusion
from .storage import require_workspace, safe_output_path


def comparison(experiments: list[dict], audit: dict) -> dict:
    if not experiments or len(experiments) > 12:
        raise ValueError("Declare 1..12 validation variants; avoid an unbounded weight search")
    if not audit.get("valid") or audit.get("class_names") != CLASS_NAMES:
        raise ValueError("A valid dataset audit in official class order is required")
    histogram = np.asarray(audit["class_histograms"]["train"])
    if histogram.shape != (NUM_CLASSES,) or not np.issubdtype(histogram.dtype, np.integer) or np.any(histogram < 0):
        raise ValueError("Audit must contain a nonnegative 32-class training histogram")
    supported = np.flatnonzero(histogram > 0)
    rare = sorted(supported.tolist(), key=lambda i: (int(histogram[i]), i))[:8]
    rows, digest, names = [], None, set()
    for experiment in experiments:
        name = experiment["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Experiment names must be unique nonempty strings")
        names.add(name)
        score = json.loads(Path(experiment["score_report"]).read_text())
        inference = json.loads(Path(experiment["prediction_report"]).read_text())
        current_digest = score.get("manifest_sha256")
        if not current_digest or inference.get("manifest_sha256") != current_digest:
            raise ValueError("Score and inference must identify the same manifest SHA256")
        if not score.get("prediction_set_sha256") or score["prediction_set_sha256"] != inference.get("prediction_set_sha256"):
            raise ValueError("Score and timing must identify the same prediction PNG content set")
        if not score.get("validation_input_sha256") or not (
            score["validation_input_sha256"] == inference.get("split_input_sha256")
            == audit.get("validation_input_sha256")
        ):
            raise ValueError("Score, inference and audit must identify the same validation input content")
        if digest is not None and digest != current_digest:
            raise ValueError("All comparisons must use the same validation manifest")
        digest = current_digest
        if audit.get("manifest_sha256") != current_digest:
            raise ValueError("Dataset audit must identify the same manifest SHA256")
        if score.get("split") != "val" or inference.get("split") != "val":
            raise ValueError("Model selection only accepts public validation results")
        if inference.get("class_names") != CLASS_NAMES:
            raise ValueError("Inference class order differs from official IDs")
        classes = score.get("classes", [])
        if [(x.get("id"), x.get("name")) for x in classes] != list(enumerate(CLASS_NAMES)):
            raise ValueError("Scores must contain all official classes including Void=31")
        matrix = np.asarray(score["confusion_matrix"])
        if matrix.shape != (NUM_CLASSES, NUM_CLASSES):
            raise ValueError("Scores need a 32x32 dataset-level confusion matrix")
        recomputed = scores_from_confusion(matrix)
        if not np.allclose([score["miou"], score["mf1"]],
                           [recomputed["miou"], recomputed["mf1"]], atol=1e-12, rtol=0):
            raise ValueError("Reported score disagrees with official macro metric")
        if not np.allclose([x["iou"] for x in classes], recomputed["iou"], atol=1e-12, rtol=0):
            raise ValueError("Reported class IoUs disagree with the confusion matrix")
        if score.get("images") != inference.get("num_predictions") or score.get("images") != audit["counts"]["val"]:
            raise ValueError("Validation image counts disagree")
        training_runs = []
        for run_dir in experiment.get("run_dirs", []):
            directory = Path(run_dir)
            metadata = json.loads((directory / "metadata.json").read_text())
            summary = json.loads((directory / "summary.json").read_text())
            if metadata["data"]["manifest_sha256"] != digest:
                raise ValueError("Training run used a different data manifest")
            if not metadata["data"].get("input_sha256") or metadata["data"]["input_sha256"] != audit.get("input_sha256"):
                raise ValueError("Training input content differs from the dataset audit")
            matching = [p for p in inference.get("checkpoint_provenance", [])
                        if Path(p["path"]).resolve().parent == directory.resolve()
                        and p["config_sha256"] == metadata["config_sha256"]]
            if not matching:
                raise ValueError("Training metadata does not match an inference checkpoint config and run path")
            training_runs.append({"run_dir": str(directory.resolve()), "seed": metadata["seed"],
                                  "model_initialization": metadata["model_initialization"],
                                  "summary": summary, "config_sha256": metadata["config_sha256"]})
        rows.append({
            "name": name, "miou_percent": recomputed["miou"] * 100,
            "mf1_percent": recomputed["mf1"] * 100,
            "class_iou_percent": [v * 100 for v in recomputed["iou"]],
            "rare_class_miou_percent": float(np.mean([recomputed["iou"][i] for i in rare])) * 100 if rare else None,
            "inference_seconds": inference.get("runtime", {}).get("prediction_seconds"),
            "peak_inference_vram_gib": inference.get("peak_allocated_gib"),
            "inference": inference, "training_runs": training_runs,
            "score_report": str(Path(experiment["score_report"]).resolve()),
            "prediction_report": str(Path(experiment["prediction_report"]).resolve()),
        })
    rows.sort(key=lambda row: -row["miou_percent"])
    return {"manifest_sha256": digest, "split": "val", "classes": CLASS_NAMES,
            "rare_class_ids": rare, "rare_class_definition": "8 least frequent nonzero training classes; ties by ID",
            "experiments": rows,
            "selection": "Ranked by official mIoU; inspect class IoUs, seed variation, memory and runtime before selection"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", required=True, help="JSON list of name, score_report, prediction_report, run_dirs")
    parser.add_argument("--audit", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    output = safe_output_path(args.report, require_workspace())
    if output.exists():
        raise FileExistsError(output)
    result = comparison(json.loads(Path(args.experiments).read_text()),
                        json.loads(Path(args.audit).read_text()))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for row in result["experiments"]:
        print(f"{row['name']}: mIoU={row['miou_percent']:.4f}% mF1={row['mf1_percent']:.4f}%")
    print(output)


if __name__ == "__main__":
    main()
