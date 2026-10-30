"""Paired image bootstrap for two validation prediction sets.

Each replicate draws the validation images with replacement, sums their
confusion matrices and recomputes the official 32-class means. Baseline and
candidate always use the same drawn images, so a replicate difference shows
how much a measured gain depends on which images were sampled. It does not
correct for model selection on the same images or for spatial correlation
between nearby panoramas.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .data import CLASS_NAMES, NUM_CLASSES, load_mask, prediction_set_sha256, read_manifest
from .metrics import confusion_matrix
from .storage import require_workspace, safe_output_path

DEFAULT_SEED = 20261028
DEFAULT_SAMPLES = 5000
BATCH = 64
STATISTIC = "32-class metric computed from sum of sampled image confusion matrices, absent classes zero"
LIMITATIONS = [
    "Image resampling does not establish spatially independent generalization.",
    "The few-pixel classes may disappear in bootstrap replicates; they still contribute zero under the official metric.",
    "Intervals describe sampling variability of these predictions and public images; they do not correct model-selection bias.",
]


def image_confusions(records: list[dict[str, str]], predictions: str | Path) -> np.ndarray:
    """Return one int64 confusion matrix per record, in manifest order."""
    predictions = Path(predictions)
    entries = list(predictions.iterdir())
    if {entry.name for entry in entries} != {record["id"] + ".png" for record in records}:
        raise ValueError(f"{predictions} must contain exactly one PNG per image in the selected split")
    if any(not entry.is_file() or entry.is_symlink() for entry in entries):
        raise ValueError(f"{predictions} must contain flat regular PNG files")
    rows = []
    for record in records:
        with Image.open(predictions / (record["id"] + ".png")) as image:
            prediction = np.asarray(image)
            if image.format != "PNG" or prediction.ndim != 2:
                raise ValueError("Predictions must be single-channel class-ID PNGs")
        rows.append(confusion_matrix(load_mask(record["mask"]), prediction, NUM_CLASSES))
    return np.stack(rows)


def class_scores(matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """IoU and F1 per class for one confusion matrix or a stack of them."""
    true_positive = np.diagonal(matrices, axis1=-2, axis2=-1).astype(np.float64)
    truth = matrices.sum(axis=-1, dtype=np.int64)
    predicted = matrices.sum(axis=-2, dtype=np.int64)
    union = truth + predicted - true_positive
    iou = np.divide(true_positive, union, out=np.zeros_like(true_positive), where=union > 0)
    denominator = truth + predicted
    f1 = np.divide(2 * true_positive, denominator, out=np.zeros_like(true_positive), where=denominator > 0)
    return iou, f1


def class_errors(per_image: np.ndarray) -> list[dict]:
    """Class scores with pixel counts and the largest confusions in both directions."""
    total = per_image.sum(axis=0)
    iou, f1 = class_scores(total)
    diagonal = np.diag(total)
    truth, predicted = total.sum(axis=1), total.sum(axis=0)
    rows = []
    for index, name in enumerate(CLASS_NAMES):
        missed = total[index].copy()
        missed[index] = 0
        incoming = total[:, index].copy()
        incoming[index] = 0
        rows.append({
            "id": index, "name": name, "iou": float(iou[index]), "f1": float(f1[index]),
            "true_positive_pixels": int(diagonal[index]), "target_pixels": int(truth[index]),
            "prediction_pixels": int(predicted[index]),
            "false_negative_pixels": int(truth[index] - diagonal[index]),
            "false_positive_pixels": int(predicted[index] - diagonal[index]),
            "precision": float(diagonal[index] / predicted[index]) if predicted[index] else 0.0,
            "recall": float(diagonal[index] / truth[index]) if truth[index] else 0.0,
            "images_with_truth": int((per_image[:, index, :].sum(axis=1) > 0).sum()),
            "images_with_prediction": int((per_image[:, :, index].sum(axis=1) > 0).sum()),
            "top_missed_to": [{"id": int(j), "name": CLASS_NAMES[j], "pixels": int(missed[j])}
                              for j in np.argsort(missed)[::-1][:5] if missed[j] > 0],
            "top_false_positives_from": [{"id": int(j), "name": CLASS_NAMES[j], "pixels": int(incoming[j])}
                                         for j in np.argsort(incoming)[::-1][:5] if incoming[j] > 0],
        })
    return rows


def _interval(values: list[np.ndarray]) -> dict[str, float]:
    values = np.concatenate(values)
    return {"percentile_2_5": float(np.quantile(values, 0.025)), "percentile_97_5": float(np.quantile(values, 0.975)),
            "median": float(np.median(values))}


def paired_bootstrap(
    baseline: np.ndarray, candidate: np.ndarray | None = None, *,
    samples: int = DEFAULT_SAMPLES, seed: int = DEFAULT_SEED,
) -> dict:
    """Bootstrap the baseline metric and, when given, the paired candidate difference."""
    if baseline.ndim != 3 or baseline.shape[1:] != (NUM_CLASSES, NUM_CLASSES) or len(baseline) == 0:
        raise ValueError("Expected a nonempty stack of 32x32 confusion matrices")
    if candidate is not None and candidate.shape != baseline.shape:
        raise ValueError("Baseline and candidate must cover the same images")
    if not 100 <= samples <= 20000:
        raise ValueError("Use 100 to 20000 bootstrap samples")
    rng = np.random.default_rng(seed)
    images = len(baseline)
    boot_iou, boot_f1, boot_delta, boot_class_delta = [], [], [], []
    for start in range(0, samples, BATCH):
        indices = rng.integers(0, images, size=(min(BATCH, samples - start), images))
        iou, f1 = class_scores(baseline[indices].sum(axis=1))
        boot_iou.append(iou.mean(axis=1))
        boot_f1.append(f1.mean(axis=1))
        if candidate is not None:
            other_iou, _ = class_scores(candidate[indices].sum(axis=1))
            boot_delta.append((other_iou - iou).mean(axis=1))
            boot_class_delta.append(other_iou - iou)
    iou, f1 = class_scores(baseline.sum(axis=0))
    result = {
        "miou": float(iou.mean()), "mf1": float(f1.mean()), "classes": class_errors(baseline),
        "bootstrap": {"seed": seed, "samples": samples, "statistic": STATISTIC,
                      "miou_distribution": _interval(boot_iou), "mf1_distribution": _interval(boot_f1)},
    }
    if candidate is not None:
        candidate_iou, candidate_f1 = class_scores(candidate.sum(axis=0))
        class_delta = np.concatenate(boot_class_delta)
        result["candidate"] = {
            "miou": float(candidate_iou.mean()), "mf1": float(candidate_f1.mean()),
            "delta_miou": float(candidate_iou.mean() - iou.mean()),
            "paired_bootstrap_delta_miou": _interval(boot_delta),
            "bootstrap_positive_fraction": float((np.concatenate(boot_delta) > 0).mean()),
            "class_delta_bootstrap_percentile_2_5": np.quantile(class_delta, 0.025, axis=0).tolist(),
            "class_delta_bootstrap_percentile_97_5": np.quantile(class_delta, 0.975, axis=0).tolist(),
            "class_delta_iou": (candidate_iou - iou).tolist(),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--baseline-report", help="palmcity.evaluate report that must match the baseline PNGs")
    parser.add_argument("--candidate-predictions")
    parser.add_argument("--report", required=True, help="New JSON path inside PALMCITY_WORKSPACE")
    parser.add_argument("--matrices", help="Optional new NPZ path for the per-image confusion matrices")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    try:
        workspace = require_workspace()
        report_path = safe_output_path(args.report, workspace)
        matrix_path = safe_output_path(args.matrices, workspace) if args.matrices else None
        for path in (report_path, matrix_path):
            if path is not None and path.exists():
                raise ValueError(f"Refusing to overwrite: {path}")
        records = read_manifest(args.manifest)[args.split]
        baseline = image_confusions(records, args.baseline_predictions)
        baseline_digest = prediction_set_sha256(args.baseline_predictions, records)
        if args.baseline_report:
            published = json.loads(Path(args.baseline_report).read_text(encoding="utf-8"))
            if (published.get("prediction_set_sha256") != baseline_digest
                    or published.get("confusion_matrix") != baseline.sum(axis=0).tolist()):
                raise ValueError("Baseline report does not match the baseline predictions")
        candidate = image_confusions(records, args.candidate_predictions) if args.candidate_predictions else None
        result = paired_bootstrap(baseline, candidate, samples=args.samples, seed=args.seed)
        report = {"kind": "public_val_class_error_and_image_bootstrap", "images": len(records),
                  "baseline_predictions": str(args.baseline_predictions), "baseline_prediction_set_sha256": baseline_digest,
                  **{key: result[key] for key in ("miou", "mf1", "classes", "bootstrap")}, "limitations": LIMITATIONS}
        if candidate is not None:
            report["candidate"] = {"predictions": str(args.candidate_predictions),
                                   "prediction_set_sha256": prediction_set_sha256(args.candidate_predictions, records),
                                   **result["candidate"]}
        if matrix_path is not None:
            arrays = {"ids": np.array([record["id"] for record in records]), "baseline": baseline}
            if candidate is not None:
                arrays["candidate"] = candidate
            matrix_path.parent.mkdir(parents=True, exist_ok=True)
            with matrix_path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            report["per_image_confusions"] = str(matrix_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("x", encoding="utf-8") as destination:
            json.dump(report, destination, indent=2, allow_nan=False)
            destination.write("\n")
        summary = f"baseline mIoU={100 * report['miou']:.2f}%"
        if candidate is not None:
            delta = report["candidate"]["paired_bootstrap_delta_miou"]
            summary += (f", candidate {100 * report['candidate']['miou']:.2f}%, difference "
                        f"{100 * report['candidate']['delta_miou']:+.2f} (95% {100 * delta['percentile_2_5']:+.2f} to "
                        f"{100 * delta['percentile_97_5']:+.2f}), positive share "
                        f"{report['candidate']['bootstrap_positive_fraction']:.3f}")
        print(f"{summary}; {report_path}")
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
