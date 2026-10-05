"""Predict class-index PNGs using one checkpoint or a probability ensemble."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .data import CLASS_NAMES, data_input_sha256, prediction_set_sha256, read_manifest
from .inference import InferenceOptions, inference_plans, panorama_probabilities
from .inference import probabilities as probabilities  # Preserve the existing public import.
from .models import NUM_CLASSES, PREPROCESSING, amp_settings, build_model, image_tensor, select_device
from .storage import require_free_space, require_workspace, safe_output_path
from .training_state import code_identity, dependency_versions, file_sha256, json_sha256


def load_checkpoint(path: str | Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load tensor-only checkpoints; never fall back to unrestricted pickle loading."""
    path = Path(path).resolve(strict=True)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError("This pipeline needs PyTorch with torch.load(weights_only=True) support") from error
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    if checkpoint.get("class_names") != list(CLASS_NAMES):
        raise ValueError(f"Checkpoint class order differs from the official 32-class order: {path}")
    if checkpoint.get("preprocessing") != PREPROCESSING:
        raise ValueError(f"Checkpoint RGB normalization differs from this pipeline: {path}")
    config = checkpoint.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError(f"Missing checkpoint model config: {path}")
    if config.get("class_names") != list(CLASS_NAMES) or config.get("preprocessing") != PREPROCESSING:
        raise ValueError(f"Inconsistent checkpoint metadata: {path}")
    if int(config["model"].get("classes", NUM_CLASSES)) != NUM_CLASSES:
        raise ValueError(f"Checkpoint model does not output 32 classes: {path}")
    image_size = config.get("image_size", [])
    if len(image_size) != 2 or not all(isinstance(value, int) for value in image_size) or image_size[0] <= 0 or image_size[1] != 2 * image_size[0]:
        raise ValueError(f"Checkpoint image_size must preserve the 2:1 panorama: {path}")
    # The state dict already contains encoder weights; construction must never download them.
    from .models import inference_model_config

    model_config = inference_model_config(config["model"])
    model = build_model(model_config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, config


def predict(
    checkpoint_paths: list[str | Path], manifest_path: str | Path, output_dir: str | Path,
    *, weights: list[float] | None = None, split: str = "test", device_name: str = "cuda:0",
    hflip_tta: bool = False, amp: bool = True,
    scales: list[float] | tuple[float, ...] = (1.0,), window_size: tuple[int, int] | None = None,
    overlap: float = 0.5, max_scaled_pixels: int = 2_097_152, max_tiles: int = 512,
    time_limit_seconds: float | None = None,
) -> Path:
    started = time.perf_counter()
    if time_limit_seconds is not None and (not math.isfinite(time_limit_seconds) or time_limit_seconds <= 0):
        raise ValueError("time_limit_seconds must be finite and positive")
    options = InferenceOptions(tuple(scales), window_size, overlap, max_scaled_pixels, max_tiles)
    workspace = require_workspace()
    if not checkpoint_paths:
        raise ValueError("Supply at least one checkpoint")
    if weights is None:
        weights = [1.0] * len(checkpoint_paths)
    if len(weights) != len(checkpoint_paths) or not all(math.isfinite(value) and value >= 0 for value in weights) or sum(weights) <= 0:
        raise ValueError("Ensemble weights must be finite, nonnegative, match checkpoint count and have a positive sum")
    weight_sum = sum(weights)
    if not math.isfinite(weight_sum):
        raise ValueError("The ensemble weight sum must be finite")
    normalized_weights = [value / weight_sum for value in weights]
    manifest = read_manifest(manifest_path)
    records = manifest.get(split, [])
    if not records:
        raise ValueError(f"Manifest has no records in split {split!r}")
    device = select_device(device_name)
    use_amp, amp_dtype = amp_settings(device, amp)
    output = safe_output_path(output_dir, workspace)
    metadata_path = safe_output_path(output.parent / f"{output.name}.json", workspace)
    if output.exists():
        raise FileExistsError(f"Prediction directory already exists: {output}")
    if metadata_path.exists():
        raise FileExistsError(f"Prediction metadata already exists: {metadata_path}")
    # Models remain in host RAM. For ensembles, only one model occupies GPU memory at a time.
    loaded = [load_checkpoint(path) for path in checkpoint_paths]
    plans = [inference_plans(tuple(config["image_size"]), options) for _, config in loaded]
    checkpoint_provenance = [
        {"path": str(Path(path).resolve()), "sha256": file_sha256(path),
         "config": config, "config_sha256": json_sha256(config), "inference_plans": plan}
        for path, (_, config), plan in zip(checkpoint_paths, loaded, plans)
    ]
    provenance = {"manifest_sha256": file_sha256(manifest_path), "code": code_identity(),
                  "dependencies": dependency_versions()}
    split_digest = data_input_sha256(manifest, splits=(split,))
    ids = [record["id"] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Prediction IDs must be unique within the split")
    for record in records:
        if Path(record["id"]).name != record["id"] or record["id"] in {".", ".."}:
            raise ValueError("Prediction IDs must be plain filename stems")
    estimated_bytes = 0
    for record in records:
        with Image.open(record["image"]) as image:
            if image.width != 2 * image.height:
                raise ValueError("Prediction images must preserve the 2:1 panorama aspect ratio")
            if image.width * image.height > max_scaled_pixels:
                raise ValueError("Prediction image exceeds max_scaled_pixels")
            estimated_bytes += image.width * image.height
    require_free_space(workspace, gib=max(0.01, 1.5 * estimated_bytes / 1024**3))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    prediction_started = time.perf_counter()
    image_timings = []
    if len(loaded) == 1:
        loaded[0][0].to(device)
    for index, record in enumerate(records, start=1):
        if time_limit_seconds is not None and time.perf_counter() - started >= time_limit_seconds:
            raise TimeoutError("Prediction time limit reached before the next image; partial PNGs were preserved")
        image_started = time.perf_counter()
        model_timings = []
        with Image.open(record["image"]) as image:
            image = image.convert("RGB")
            native_size = (image.height, image.width)
            accumulated = torch.zeros((NUM_CLASSES, *native_size), dtype=torch.float32)
            for (model, config), weight in zip(loaded, normalized_weights):
                if weight == 0:
                    continue
                model_started = time.perf_counter()
                inputs = image_tensor(image, native_size)
                if len(loaded) > 1:
                    model.to(device)
                accumulated.add_(panorama_probabilities(
                    model, inputs, native_size, device=device, use_amp=use_amp,
                    amp_dtype=amp_dtype, hflip_tta=hflip_tta, options=options,
                    base_size=tuple(config["image_size"]),
                ), alpha=weight)
                if len(loaded) > 1:
                    model.to("cpu")
                del inputs
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                model_timings.append(time.perf_counter() - model_started)
        prediction = accumulated.argmax(dim=0).numpy().astype(np.uint8)
        Image.fromarray(prediction).save(output / f"{record['id']}.png")
        image_timings.append({"id": record["id"], "seconds": time.perf_counter() - image_started,
                              "active_model_seconds": model_timings})
        print(f"Predicted {index}/{len(records)}: {record['id']}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    prediction_seconds = time.perf_counter() - prediction_started
    peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    prediction_digest = prediction_set_sha256(output, records)
    metadata = {
        "checkpoints": [str(Path(path).resolve()) for path in checkpoint_paths],
        "weights": normalized_weights, "horizontal_flip_tta": hflip_tta, "split": split,
        "num_predictions": len(records), "class_names": list(CLASS_NAMES),
        "manifest_sha256": provenance["manifest_sha256"],
        "split_input_sha256": split_digest,
        "prediction_set_sha256": prediction_digest,
        "checkpoint_sha256": [item["sha256"] for item in checkpoint_provenance],
        "inference_seconds": sum(sum(item["active_model_seconds"]) for item in image_timings),
        "per_image_seconds": prediction_seconds / len(records),
        "peak_allocated_gib": peak_allocated / 1024**3 if peak_allocated is not None else None,
        "peak_reserved_gib": peak_reserved / 1024**3 if peak_reserved is not None else None,
        "preprocessing": PREPROCESSING, "inference": asdict(options),
        "checkpoint_provenance": checkpoint_provenance, "provenance": provenance,
        "device": str(device), "amp": use_amp, "amp_dtype": str(amp_dtype) if use_amp else None,
        "runtime": {"total_seconds": time.perf_counter() - started,
                    "prediction_seconds": prediction_seconds,
                    "seconds_per_image": prediction_seconds / len(records),
                    "images_per_second": len(records) / prediction_seconds,
                    "per_image": image_timings,
                    "timing_scope": "prediction includes RGB preprocessing, CPU/GPU model transfer, inference and PNG writing; total also includes checkpoint loading and provenance hashing"},
        "memory": {"peak_allocated_bytes": peak_allocated,
                   "peak_reserved_bytes": peak_reserved,
                   "probabilities": "image-at-a-time CPU accumulation; no dataset probability files"},
    }
    # Keep PNG prediction folders clean for direct submission packaging.
    with metadata_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs="+", required=True, help="Trusted local pipeline checkpoints")
    parser.add_argument("--weights", nargs="+", type=float, help="Validation-selected ensemble weights, in checkpoint order")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True, help="A new path inside PALMCITY_WORKSPACE")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hflip-tta", action="store_true")
    parser.add_argument("--scales", nargs="+", type=float, default=[1.0], help="1 to 8 factors relative to each checkpoint input size; probability TTA")
    parser.add_argument("--window-size", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"), help="Overlapping panorama windows with 2:1 aspect ratio, e.g. 512 1024")
    parser.add_argument("--overlap", type=float, default=0.5, help="Sliding-window overlap fraction in [0, 1)")
    parser.add_argument("--max-scaled-pixels", type=int, default=2_097_152, help="Per-scale CPU probability-map size bound")
    parser.add_argument("--max-tiles", type=int, default=512, help="Maximum tiles per image scale")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--time-limit-seconds", type=float, help="Stop before the next image when elapsed time reaches this limit; preserve partial outputs")
    args = parser.parse_args(argv)
    output = predict(args.checkpoint, args.manifest, args.output_dir, weights=args.weights,
                     split=args.split, device_name=args.device, hflip_tta=args.hflip_tta, amp=not args.no_amp,
                     scales=args.scales, window_size=tuple(args.window_size) if args.window_size else None,
                     overlap=args.overlap, max_scaled_pixels=args.max_scaled_pixels, max_tiles=args.max_tiles,
                     time_limit_seconds=args.time_limit_seconds)
    print(f"Prediction directory: {output}")


if __name__ == "__main__":
    main()
