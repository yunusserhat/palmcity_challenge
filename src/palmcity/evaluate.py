"""Score saved train/validation PNGs with the published PalmCity metric."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .data import CLASS_NAMES, NUM_CLASSES, data_input_sha256, load_mask, prediction_set_sha256, read_manifest
from .metrics import confusion_matrix, scores_from_confusion
from .storage import require_workspace, safe_output_path


def evaluate_predictions(manifest: dict, predictions: str | Path, split: str = "val") -> dict:
    if split not in {"train", "val"}:
        raise ValueError("Evaluation requires the public train or val labels.")
    records = manifest[split]
    if not records:
        raise ValueError(f"No {split} records in manifest.")
    predictions = Path(predictions)
    entries = list(predictions.iterdir())
    expected = {record["id"] + ".png" for record in records}
    if {entry.name for entry in entries} != expected:
        raise ValueError("Predictions must contain exactly one PNG per image in the selected split.")
    if any(not entry.is_file() or entry.is_symlink() for entry in entries):
        raise ValueError("Predictions must be flat regular PNG files.")
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for record in records:
        target = load_mask(record["mask"])
        with Image.open(predictions / (record["id"] + ".png")) as image:
            prediction = np.asarray(image)
            if image.format != "PNG" or prediction.ndim != 2 or "transparency" in image.info:
                raise ValueError("Predictions must be opaque, single-channel class-ID PNGs.")
        matrix += confusion_matrix(target, prediction, NUM_CLASSES)
    scores = scores_from_confusion(matrix)
    return {
        "split": split, "images": len(records), "miou": scores["miou"], "mf1": scores["mf1"],
        "miou_percent": 100 * scores["miou"], "mf1_percent": 100 * scores["mf1"],
        "classes": [
            {"id": index, "name": name, "iou": scores["iou"][index],
             "f1": scores["f1"][index], "target_pixels": int(matrix[index].sum())}
            for index, name in enumerate(CLASS_NAMES)
        ],
        "confusion_matrix": matrix.tolist(),
        "prediction_set_sha256": prediction_set_sha256(predictions, records),
        "validation_input_sha256": data_input_sha256(manifest, splits=(split,)),
        "metric": "dataset-level macro mean of all 32 classes; absent classes score zero",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    try:
        workspace = require_workspace()
        report_path = safe_output_path(args.report, workspace)
        if report_path.exists():
            raise ValueError(f"Refusing to overwrite report: {report_path}")
        report = evaluate_predictions(read_manifest(args.manifest), args.predictions, args.split)
        report["manifest_sha256"] = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("x", encoding="utf-8") as destination:
            json.dump(report, destination, indent=2, allow_nan=False)
            destination.write("\n")
        print(f"mIoU={report['miou_percent']:.4f}%, mF1={report['mf1_percent']:.4f}%; {report_path}")
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
