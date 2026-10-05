"""Dataset-level PalmCity metrics, matching the official scorer.

All 32 classes, including Void (31), enter both means. Absent classes have
zero IoU/F1. Values here are fractions; Codabench displays percentages.
https://github.com/PalmCityDataset/palmcity/blob/main/evaluation/scoring.py
"""

from __future__ import annotations

import numpy as np


def confusion_matrix(target: np.ndarray, prediction: np.ndarray, num_classes: int = 32) -> np.ndarray:
    """Return int64 counts with target on rows and prediction on columns."""
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")
    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.shape != prediction.shape:
        raise ValueError(f"Mask shape mismatch: {target.shape} != {prediction.shape}")
    for label, values in (("target", target), ("prediction", prediction)):
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{label} must contain integer class IDs")
        if np.any((values < 0) | (values >= num_classes)):
            raise ValueError(f"{label} contains class IDs outside 0..{num_classes - 1}")
    encoded = num_classes * target.astype(np.int64, copy=False).ravel() + prediction.astype(np.int64, copy=False).ravel()
    return np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def scores_from_confusion(confusion: np.ndarray) -> dict[str, float | list[float]]:
    """Compute macro scores from a sum of per-image confusion matrices."""
    confusion = np.asarray(confusion)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1] or confusion.shape[0] == 0:
        raise ValueError("Confusion matrix must be a nonempty square matrix")
    if not np.issubdtype(confusion.dtype, np.integer) or np.any(confusion < 0):
        raise ValueError("Confusion matrix must contain nonnegative integer counts")
    true_positive = np.diag(confusion).astype(np.float64)
    target_count = confusion.sum(axis=1).astype(np.float64)
    prediction_count = confusion.sum(axis=0).astype(np.float64)
    iou_denominator = target_count + prediction_count - true_positive
    f1_denominator = target_count + prediction_count
    iou = np.divide(true_positive, iou_denominator, out=np.zeros_like(true_positive), where=iou_denominator > 0)
    f1 = np.divide(2.0 * true_positive, f1_denominator, out=np.zeros_like(true_positive), where=f1_denominator > 0)
    return {"miou": float(iou.mean()), "mf1": float(f1.mean()), "iou": iou.tolist(), "f1": f1.tolist()}
