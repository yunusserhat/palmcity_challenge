"""Export trusted local EoMT checkpoints as unchanged inference-only HF safe weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from .data import CLASS_NAMES, PALETTE
from .models import PREPROCESSING
from .predict import load_checkpoint
from .storage import managed_directory, require_free_space, require_workspace, safe_output_path


def export_checkpoint(checkpoint: str | Path, output: str | Path) -> dict:
    """Copy float32 tensors exactly; retain no training state or private filesystem paths."""
    workspace = require_workspace()
    output = safe_output_path(output, workspace)
    if output.exists():
        raise FileExistsError(f"Export directory already exists: {output}")
    checkpoint = Path(checkpoint).resolve(strict=True)
    require_free_space(workspace, gib=max(0.01, checkpoint.stat().st_size * 1.5 / 1024**3))
    model, config = load_checkpoint(checkpoint)
    if config["model"].get("architecture") != "eomt_dinov3" or not hasattr(model, "network"):
        raise ValueError("This release exporter supports EoMT with DINOv3 only")
    output = managed_directory(output)
    model.network.save_pretrained(output, safe_serialization=True, max_shard_size="5GB")
    weight_file = output / "model.safetensors"
    # Export verification compares every tensor and dtype, including all persistent buffers.
    with safe_open(weight_file, framework="pt", device="cpu") as exported:
        state = model.network.state_dict()
        if set(exported.keys()) != set(state):
            raise RuntimeError("Export changed the set of model tensors")
        for name, tensor in state.items():
            actual = exported.get_tensor(name)
            if actual.dtype != tensor.dtype or not torch.equal(actual, tensor.cpu()):
                raise RuntimeError(f"Export changed tensor {name}")
    metadata = {
        "format_version": 1, "architecture": "eomt_dinov3", "seed": config["seed"],
        "class_names": list(CLASS_NAMES), "palette": PALETTE.tolist(),
        "image_size": config["image_size"], "preprocessing": PREPROCESSING,
        "semantic_classes": 32, "void_class_id": 31,
        "query_no_object_id": 32, "rectangular_patch_grid": True,
        "probability_conversion": "softmax(query_classes)[..., :32] times sigmoid(query_masks), summed over queries and normalized over classes",
        "source": {"repo": config["pretrained_source"]["repo"],
                   "revision": config["pretrained_source"]["revision"]},
    }
    (output / "palmcity_inference.json").write_text(json.dumps(metadata, indent=2) + "\n")
    processor = {"image_processor_type": "EomtImageProcessor", "do_resize": True,
                 "size": {"height": config["image_size"][0], "width": config["image_size"][1]},
                 "resample": 2, "do_rescale": True, "rescale_factor": 1 / 255.0,
                 "do_normalize": True, "image_mean": PREPROCESSING["mean"],
                 "image_std": PREPROCESSING["std"], "do_pad": False,
                 "do_split_image": False, "do_reduce_labels": False,
                 "ignore_index": 255}
    # The project helper applies exact native-image normalization and scale sampling.
    # This plain metadata is also readable without importing custom model code.
    (output / "preprocessor_config.json").write_text(json.dumps(processor, indent=2) + "\n")
    return {"seed": config["seed"], "tensor_count": len(state),
            "weight_bytes": weight_file.stat().st_size, "tensor_equality": True,
            "class_names": list(CLASS_NAMES), "image_size": config["image_size"],
            "preprocessing": PREPROCESSING}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True, help="New release folder inside the workspace")
    args = parser.parse_args(argv)
    workspace = require_workspace()
    output = safe_output_path(args.output_dir, workspace)
    if output.exists():
        raise FileExistsError(f"Export directory already exists: {output}")
    required = sum(Path(path).stat().st_size for path in args.checkpoint) * 1.5 / 1024**3
    require_free_space(workspace, gib=max(1, required))
    # Derive seed from filename-independent checkpoint metadata inside each export.
    managed_directory(output)
    summaries = []
    for index, path in enumerate(args.checkpoint):
        temporary = output / f"export-{index}"
        summary = export_checkpoint(path, temporary)
        target = output / f"seed{summary['seed']}"
        if target.exists():
            raise ValueError(f"Repeated seed {summary['seed']} in the supplied checkpoints")
        temporary.rename(target)
        summaries.append(summary)
    metadata = {"format_version": 1, "class_names": list(CLASS_NAMES),
                "palette": PALETTE.tolist(), "preprocessing": PREPROCESSING,
                "image_size": summaries[0]["image_size"],
                "seeds": [summary["seed"] for summary in summaries],
                "single_model_seed": 2026,
                "challenge_inference": {"scales": [0.75, 1.0, 1.25], "horizontal_flip": True,
                                        "ensemble": "equal arithmetic mean of class probabilities"}}
    (output / "palmcity_inference.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"exports": summaries}, indent=2))


if __name__ == "__main__":
    main()
