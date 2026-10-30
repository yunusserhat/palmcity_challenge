"""Training-mask image sampling; no split discovery or label modification.

Pass only ``manifest["train"]`` to :func:`training_sample_weights`. The caller
owns the split: a record without an explicit split has no provenance that this
module can independently verify. Image files and other manifest splits are
never opened. Excluding a class from rarity calculation does not ignore that
class in training or evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from .data import NUM_CLASSES, load_mask


DEFAULT_EXCLUDED_CLASS_IDS = (0, 19, 26, 31)
_CONFIG_KEYS = {"power", "max_weight", "min_pixels", "excluded_class_ids"}


def _sampling_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping) or set(config) - _CONFIG_KEYS:
        raise ValueError(f"Sampling config must contain only {sorted(_CONFIG_KEYS)}")
    values: dict[str, Any] = {}
    for key, default in (("power", 0.5), ("max_weight", 5.0)):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Sampling {key} must be a finite number")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"Sampling {key} must be a finite number")
        values[key] = value
    if not 0 <= values["power"] <= 1:
        raise ValueError("Sampling power must be between 0 and 1")
    if values["max_weight"] < 1:
        raise ValueError("Sampling max_weight must be at least 1")
    min_pixels = config.get("min_pixels", 16)
    if type(min_pixels) is not int or min_pixels < 1:
        raise ValueError("Sampling min_pixels must be a positive integer")
    values["min_pixels"] = min_pixels
    excluded = config.get("excluded_class_ids", DEFAULT_EXCLUDED_CLASS_IDS)
    if not isinstance(excluded, (list, tuple)) or any(
        type(value) is not int or not 0 <= value < NUM_CLASSES for value in excluded
    ):
        raise ValueError("Sampling excluded_class_ids must list integer class IDs 0..31")
    if len(set(excluded)) != len(excluded):
        raise ValueError("Sampling excluded_class_ids must not contain duplicates")
    values["excluded_class_ids"] = sorted(excluded)
    return values


def training_sample_weights(
    records: Sequence[Mapping[str, str]], config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return CPU float64 weights and JSON-safe training-only metadata.

    ``config`` accepts ``power`` (default 0.5), ``max_weight`` (5),
    ``min_pixels`` (16), and ``excluded_class_ids`` ([0, 19, 26, 31]). For N
    training masks, a class appears in an image only when it has at least
    ``min_pixels`` native-resolution pixels. Its rarity is
    ``(N / class_image_count) ** power``. An image receives the maximum
    eligible rarity present, bounded to [1, max_weight]. Images without an
    eligible class receive 1. Weights are relative to this baseline, not
    normalized to mean 1; the sampler normalizes probabilities internally.

    Labels, including Void=31, remain untouched. Only the supplied masks are
    read; callers must pass training records rather than a full manifest or
    concatenate validation/test records. Explicit non-training split tags
    are rejected. Metadata describes the exact record order used by weights.
    """
    options = _sampling_config(config)
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise ValueError("Sampling requires a nonempty sequence of training records")
    ids: list[str] = []
    mask_paths: list[Path] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Sampling records must be training record mappings")
        if record.get("split", "train") != "train":
            raise ValueError("Sampling accepts only training records")
        identifier, mask = record.get("id"), record.get("mask")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("Sampling training records require nonempty string IDs")
        if not isinstance(mask, str) or not mask:
            raise ValueError(f"Sampling training record {identifier}: missing mask path")
        path = Path(mask)
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"Sampling training record {identifier}: missing absolute mask file {path}")
        ids.append(identifier)
        mask_paths.append(path)
    if len(set(ids)) != len(ids):
        raise ValueError("Sampling training record IDs must be unique")

    presence = np.zeros((len(records), NUM_CLASSES), dtype=bool)
    pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for index, path in enumerate(mask_paths):
        # load_mask enforces official 0..31 IDs and never changes source files.
        counts = np.bincount(load_mask(path).reshape(-1), minlength=NUM_CLASSES)
        presence[index] = counts >= options["min_pixels"]
        pixel_counts += counts
    image_counts = presence.sum(axis=0)
    class_weights = np.ones(NUM_CLASSES, dtype=np.float64)
    eligible = image_counts > 0
    eligible[options["excluded_class_ids"]] = False
    class_weights[eligible] = np.minimum(
        (len(records) / image_counts[eligible]) ** options["power"],
        options["max_weight"],
    )
    weights = np.maximum(1.0, np.where(presence, class_weights[None, :], 1.0).max(axis=1))
    metadata = {
        "method": "maximum_inverse_training_image_frequency",
        "source": "provided_training_records_only",
        "config": options,
        "num_records": len(records),
        "record_ids": ids,
        "class_image_counts": image_counts.tolist(),
        "class_pixel_counts": pixel_counts.tolist(),
        "class_weights": class_weights.tolist(),
        "weights": weights.tolist(),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
        "replacement": True,
        "samples_per_epoch": len(records),
    }
    return torch.from_numpy(weights.copy()), metadata


def make_training_sampler(
    weights: torch.Tensor, *, generator: torch.Generator,
) -> WeightedRandomSampler:
    """Draw exactly N images with replacement using a caller-owned CPU RNG.

    Save and restore ``generator.get_state()`` with training checkpoints. A
    DataLoader may use that same generator; set ``shuffle=False`` when passing
    this sampler. Iteration advances the generator, so restoring its state
    reproduces the next epoch's complete sequence.
    """
    if not isinstance(weights, torch.Tensor) or weights.ndim != 1 or weights.numel() == 0:
        raise ValueError("Sampling weights must be a nonempty one-dimensional tensor")
    if weights.device.type != "cpu" or weights.is_complex():
        raise ValueError("Sampling weights must be real CPU values")
    if not torch.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("Sampling weights must be finite and strictly positive")
    if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
        raise ValueError("Sampling requires a caller-owned CPU torch.Generator")
    return WeightedRandomSampler(
        weights.detach().to(dtype=torch.float64), num_samples=weights.numel(),
        replacement=True, generator=generator,
    )
