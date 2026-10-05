"""Run released PalmCity safetensors on local panoramic images, without training data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .data import CLASS_NAMES, IMAGE_EXTENSIONS, PALETTE
from .inference import InferenceOptions, panorama_probabilities
from .models import PREPROCESSING, amp_settings, image_tensor, select_device
from .storage import managed_directory, require_free_space, require_workspace, safe_output_path
from .transformer_models import QueryTransformerModel

HUB_MODEL = "yunusserhat/palmcity-eomt-dinov3-large"
SEEDS = (42, 123, 2026)


def read_release_metadata(directory: str | Path) -> dict[str, Any]:
    """Require the exact published class IDs and preprocessing before model loading."""
    directory = Path(directory)
    metadata = json.loads((directory / "palmcity_inference.json").read_text())
    if metadata.get("format_version") != 1:
        raise ValueError("Unsupported PalmCity Hub release format")
    if metadata.get("class_names") != list(CLASS_NAMES):
        raise ValueError("Released class order differs from PalmCity's 32 classes")
    if metadata.get("preprocessing") != PREPROCESSING:
        raise ValueError("Released RGB normalization differs from the trained models")
    size = metadata.get("image_size")
    if (not isinstance(size, list) or len(size) != 2
            or not all(isinstance(value, int) and value > 0 for value in size)
            or size[1] != 2 * size[0]):
        raise ValueError("Released image_size must preserve the 2:1 panorama")
    return metadata


def resolve_release(
    model: str | Path, seeds: tuple[int, ...], *, revision: str | None = None,
    download: bool = False,
) -> Path:
    """Use a local bundle or the existing Hub cache; only explicit download permits network."""
    candidate = Path(model).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    if candidate.is_absolute() or str(model).startswith((".", "~")):
        raise FileNotFoundError(f"Local model directory does not exist: {candidate}")
    if download:
        cache = Path(os.environ.get("HF_HUB_CACHE", ""))
        if not cache.is_absolute() or not cache.is_dir():
            raise RuntimeError("An existing absolute HF_HUB_CACHE is required for downloads")
        if cache.stat().st_uid != os.getuid() or not os.access(cache, os.W_OK | os.X_OK):
            raise RuntimeError("The existing Hugging Face cache must be writable by this user")
        # Each unchanged float32 seed is about 1.18 GiB; retain room for transfer temporaries.
        require_free_space(cache, gib=1.5 * len(seeds) + 1)
        if os.environ.get("PALMCITY_WORF_PROFILE") == "1":
            if not cache.is_relative_to(Path("/scratch")):
                raise RuntimeError("Worf Hub downloads must use the existing scratch cache")
            if cache.stat().st_dev != os.stat("/scratch").st_dev:
                raise RuntimeError("The Worf Hub cache is on an unexpected filesystem")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        str(model), revision=revision, local_files_only=not download,
        allow_patterns=["palmcity_inference.json", *[
            f"seed{seed}/{filename}" for seed in seeds
            for filename in ("config.json", "model.safetensors", "palmcity_inference.json")
        ]],
    ))


def load_released_model(directory: str | Path) -> tuple[QueryTransformerModel, dict[str, Any]]:
    """Load native Transformers safe weights and apply the trained panorama adapter."""
    from transformers import EomtDinov3ForUniversalSegmentation

    directory = Path(directory)
    metadata = read_release_metadata(directory)
    config = json.loads((directory / "config.json").read_text())
    if config.get("model_type") != "eomt_dinov3":
        raise ValueError("The released architecture must be EoMT with DINOv3")
    labels = {str(index): name for index, name in enumerate(CLASS_NAMES)}
    if config.get("id2label") != labels:
        raise ValueError("Native Transformers config has a different class order")
    if not (directory / "model.safetensors").is_file():
        raise FileNotFoundError("Release needs model.safetensors; pickle weights are not loaded")
    network, loading = EomtDinov3ForUniversalSegmentation.from_pretrained(
        directory, local_files_only=True, use_safetensors=True, output_loading_info=True,
    )
    if any(loading.get(name) for name in
           ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Released model tensors are incompatible: {loading}")
    # eval disables query attention masks, and forward derives a rectangular patch grid.
    return QueryTransformerModel(network, eomt=True).eval(), metadata


@torch.inference_mode()
def released_probabilities(
    loaded: list[tuple[torch.nn.Module, dict[str, Any]]], image: Image.Image,
    *, device: torch.device, scales: tuple[float, ...] = (1.0,),
    hflip_tta: bool = False, amp: bool = True,
) -> torch.Tensor:
    """Use identical probability conversion and TTA to the measured challenge pipeline."""
    if not loaded:
        raise ValueError("Supply at least one released model")
    options = InferenceOptions(scales=scales)
    native_size = (image.height, image.width)
    inputs = image_tensor(image, native_size)
    use_amp, amp_dtype = amp_settings(device, amp)
    result = torch.zeros((len(CLASS_NAMES), *native_size), dtype=torch.float32)
    for model, metadata in loaded:
        model.to(device)
        try:
            result.add_(panorama_probabilities(
                model, inputs, native_size, device=device, use_amp=use_amp,
                amp_dtype=amp_dtype, options=options, hflip_tta=hflip_tta,
                base_size=tuple(metadata["image_size"]),
            ), alpha=1 / len(loaded))
        finally:
            # One model occupies GPU memory at a time, as in the measured submission.
            # A single model stays resident across images to avoid repeated transfers.
            if len(loaded) > 1:
                model.to("cpu")
    return result


def input_images(path: str | Path) -> list[Path]:
    path = Path(path).expanduser().resolve(strict=True)
    images = ([path] if path.is_file() else sorted(
        item for item in path.iterdir() if item.is_file()
        and item.suffix.lower() in IMAGE_EXTENSIONS))
    if not images or any(item.suffix.lower() not in IMAGE_EXTENSIONS for item in images):
        raise ValueError("Input must be an image or a directory containing images")
    if len({item.stem for item in images}) != len(images):
        raise ValueError("Input image stems must be unique to avoid overwriting predictions")
    for item in images:
        with Image.open(item) as image:
            if image.width != 2 * image.height:
                raise ValueError(f"{item.name}: the trained model expects a 2:1 panorama")
            if image.width * image.height > 2_097_152:
                raise ValueError(f"{item.name}: image exceeds the inference pixel bound")
    return images


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=HUB_MODEL, help="Hub ID or local exported bundle")
    parser.add_argument("--revision", help="Optional immutable Hub revision for exact repeatability")
    parser.add_argument("--download", action="store_true", help="Permit missing model files to download")
    parser.add_argument("--input", required=True, help="Local RGB panorama or image directory")
    parser.add_argument("--output-dir", required=True, help="New folder inside PALMCITY_WORKSPACE")
    parser.add_argument("--mode", choices=("fast", "single-tta", "challenge"), default="fast")
    parser.add_argument("--seed", choices=SEEDS, type=int, default=2026,
                        help="Single-model seed; challenge mode always uses all three")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--colorize", action="store_true", help="Also write palette visualizations")
    args = parser.parse_args(argv)
    workspace = require_workspace()
    output = safe_output_path(args.output_dir, workspace)
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    images = input_images(args.input)
    seeds = SEEDS if args.mode == "challenge" else (args.seed,)
    release = resolve_release(args.model, seeds, revision=args.revision, download=args.download)
    root_metadata = read_release_metadata(release)
    loaded = [load_released_model(release / f"seed{seed}") for seed in seeds]
    if any(metadata["image_size"] != root_metadata["image_size"] for _, metadata in loaded):
        raise ValueError("Seed preprocessing resolutions differ from the ensemble metadata")
    device = select_device(args.device)
    # Grayscale IDs plus optional RGB pictures are bounded by four bytes per input pixel.
    pixels = 0
    for item in images:
        with Image.open(item) as image:
            pixels += image.width * image.height
    require_free_space(workspace, gib=max(0.01, pixels * 6 / 1024**3))
    managed_directory(output)
    mask_dir = managed_directory(output / "masks")
    color_dir = managed_directory(output / "colorized") if args.colorize else None
    scales = (1.0,) if args.mode == "fast" else (0.75, 1.0, 1.25)
    for item in images:
        with Image.open(item) as image:
            probabilities = released_probabilities(
                loaded, image.convert("RGB"), device=device, scales=scales,
                hflip_tta=args.mode != "fast", amp=not args.no_amp,
            )
        prediction = probabilities.argmax(0).numpy().astype(np.uint8)
        Image.fromarray(prediction).save(mask_dir / f"{item.stem}.png")
        if color_dir is not None:
            Image.fromarray(PALETTE[prediction]).save(color_dir / f"{item.stem}.png")
        print(f"Predicted {item.name}", flush=True)
    summary = {"model": args.model, "revision": args.revision, "seeds": list(seeds),
               "mode": args.mode, "scales": list(scales), "horizontal_flip": args.mode != "fast",
               "class_names": CLASS_NAMES, "preprocessing": PREPROCESSING,
               "images": len(images), "device": str(device)}
    (output / "inference.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Class-ID PNGs: {mask_dir}")


if __name__ == "__main__":
    main()
