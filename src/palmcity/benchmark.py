"""Bounded, download-free synthetic GPU probes; never measure competition quality."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import time
from pathlib import Path

import torch

from .models import NUM_CLASSES, amp_settings, build_model, optimizer_parameter_groups, select_device
from .storage import require_free_space, require_workspace, safe_output_path


def idle_gpu(index: int) -> dict:
    """Check the selected physical GPU before the short shared-machine probe."""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Use physical cuda:N with CUDA_VISIBLE_DEVICES unset for this probe")
    info = subprocess.run(
        ["nvidia-smi", f"--id={index}", "--query-gpu=uuid,name,memory.total,memory.free",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True,
    ).stdout.strip().split(", ")
    if len(info) != 4:
        raise RuntimeError("Could not identify the selected GPU")
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    if any(line.split(",")[0].strip() == info[0] for line in processes):
        raise RuntimeError("Selected GPU has a compute process; defer the probe")
    if float(info[3]) < 0.9 * float(info[2]):
        raise RuntimeError("Selected GPU has less than 90% free memory; defer the probe")
    return {"physical_index": index, "uuid": info[0], "name": info[1],
            "total_mib": float(info[2]), "free_mib_before": float(info[3])}


def proposed_budget(report: dict, *, hours: float = 1.0, train_images: int = 497,
                    val_images: int = 84) -> dict:
    """Include estimated validation cost when translating GPU time into updates."""
    if report.get("status") != "ok":
        raise ValueError("A successful probe is required for a budget")
    if not math.isfinite(hours) or hours <= 0 or train_images <= 0 or val_images < 0:
        raise ValueError("Budget hours/counts must be finite and positive (val may be zero)")
    batch = int(report["batch_size"])
    accumulation = int(report["gradient_accumulation"])
    batches = train_images // batch if report["drop_last"] else math.ceil(train_images / batch)
    updates_per_epoch = math.ceil(batches / accumulation)
    if updates_per_epoch <= 0:
        raise ValueError("Training must contain at least one batch")
    training_seconds = float(report["training_microbatch_seconds_median"])
    inference_seconds = float(report["inference_seconds_per_image"])
    if not all(math.isfinite(x) and x > 0 for x in [training_seconds, inference_seconds]):
        raise ValueError("Probe durations must be finite and positive")
    seconds_per_update = float(report.get("training_update_seconds_median", training_seconds * accumulation))
    val_seconds = inference_seconds * val_images
    estimated_update_seconds = seconds_per_update + val_seconds / updates_per_epoch
    updates = max(1, int(hours * 3600 / estimated_update_seconds))
    return {
        "target_gpu_hours_per_run": hours, "effective_batch_size": batch * accumulation,
        "max_optimizer_steps": updates, "epochs": math.ceil(updates / updates_per_epoch),
        "estimated_updates_per_epoch": updates_per_epoch,
        "estimated_train_seconds_per_update": seconds_per_update,
        "estimated_validation_seconds_per_epoch": val_seconds,
        "estimated_total_seconds": updates * estimated_update_seconds,
        "basis": "synthetic model-only estimate; recalibrate with real data before pilots",
        "excludes": ["data loading", "augmentation", "checkpoint IO", "startup"],
        "scheduler_note": "Set max_optimizer_steps before training; retain it on resume",
    }


def probe(config_path: str | Path, report_path: str | Path, *, device_name: str = "cuda:0",
          steps: int = 3, warmup: int = 1, pilot_hours: float = 1.0) -> dict:
    if not 1 <= steps <= 10 or not 1 <= warmup <= 3:
        raise ValueError("Bounded probe needs 1..10 measured and 1..3 warmup steps")
    workspace = require_workspace()
    output = safe_output_path(report_path, workspace)
    if output.exists():
        raise FileExistsError(output)
    raw = Path(config_path).read_bytes()
    config = json.loads(raw)
    height, width = map(int, config["image_size"])
    batch = int(config["batch_size"])
    if height <= 0 or width != 2 * height or batch <= 0:
        raise ValueError("Probe must preserve a positive 2:1 panorama with a positive batch")
    if any(config["model"].get(key) for key in (
        "encoder_weights", "encoder_pretrained_path", "pretrained", "pretrained_model_name_or_path", "pretrained_backbone_name_or_path",
    )):
        raise ValueError("Synthetic probes require random initialization; pretrained is forbidden")
    device_spec = torch.device(device_name)
    if device_spec.type != "cuda" or device_spec.index is None:
        raise ValueError("Synthetic resource probes require explicit cuda:N")
    gpu = idle_gpu(device_spec.index)
    device = select_device(device_name)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    require_free_space(workspace, gib=1.0)
    torch.manual_seed(42)
    use_amp, dtype = amp_settings(device, bool(config.get("amp", True)))
    report = {
        "schema_version": 2, "status": "initializing", "synthetic": True,
        "weights": "random", "quality_metrics": None, "gpu": gpu,
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": hashlib.sha256(raw).hexdigest(), "model_config": config["model"],
        "image_size": [height, width], "batch_size": batch,
        "gradient_accumulation": int(config.get("gradient_accumulation", 1)),
        "drop_last": bool(config.get("drop_last", False)), "seed": 42,
        "warmup_steps": warmup, "measured_steps": steps,
        "amp": use_amp, "amp_dtype": str(dtype),
        "packages": {name: importlib.metadata.version(name) for name in
                     ["torch", "torchvision", "segmentation-models-pytorch", "timm", "transformers", "scipy"]},
    }
    from .train import model_training_loss
    from .training_state import code_identity

    report["code"] = code_identity()

    started = time.perf_counter()
    try:
        model = build_model(config["model"]).to(device)
        if config.get("gradient_checkpointing"):
            enable = getattr(model, "gradient_checkpointing_enable", None)
            if enable is None:
                raise ValueError("This model does not support requested gradient checkpointing")
            enable()
        parameters = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        report.update({"parameters": parameters, "trainable_parameters": trainable,
                       "fp32_weights_gib": parameters * 4 / 1024**3,
                       "fp32_adam_training_state_floor_gib": (parameters * 4 + trainable * 12) / 1024**3})
        learning_rate = float(config["learning_rate"])
        groups = optimizer_parameter_groups(model, learning_rate,
                                            float(config.get("weight_decay", 0.01)),
                                            float(config.get("backbone_lr_multiplier", 1)))
        optimizer = torch.optim.AdamW(groups, lr=learning_rate)
        report["optimizer_groups"] = [{"name": group["group_name"], "learning_rate": group["lr"],
                                       "weight_decay": group["weight_decay"]} for group in groups]
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype == torch.float16)
        images = torch.randn(batch, 3, height, width, device=device)
        targets = (torch.arange(height * width, device=device).reshape(1, height, width)
                   .repeat(batch, 1, 1) % NUM_CLASSES).long()
        model.train()
        if hasattr(model, "set_training_progress"):
            model.set_training_progress(0, 1)
        durations = []
        accumulation = report["gradient_accumulation"]
        if accumulation <= 0:
            raise ValueError("gradient_accumulation must be positive")
        for index in range(warmup + steps):
            if index == warmup:
                torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            tick = time.perf_counter()
            for _ in range(accumulation):
                with torch.autocast("cuda", dtype=dtype, enabled=use_amp):
                    loss = model_training_loss(model, images, targets, float(config.get("dice_weight", 0)))
                if not torch.isfinite(loss):
                    raise RuntimeError("Synthetic training loss is nonfinite")
                scaler.scale(loss / accumulation).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1)))
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize(device)
            if index >= warmup:
                durations.append(time.perf_counter() - tick)
        train_peak = torch.cuda.max_memory_allocated(device)
        reserved_peak = torch.cuda.max_memory_reserved(device)
        # Training peak and inference peak are independent; release optimizer/grad states.
        del optimizer, scaler, targets, loss
        model.zero_grad(set_to_none=True)
        images = images[:1].clone()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model.eval()
        inference = []
        with torch.inference_mode():
            for index in range(warmup + steps):
                torch.cuda.synchronize(device)
                tick = time.perf_counter()
                with torch.autocast("cuda", dtype=dtype, enabled=use_amp):
                    logits = model(images)
                if logits.shape[:2] != (1, NUM_CLASSES) or not torch.isfinite(logits).all():
                    raise RuntimeError("Model must emit finite dense logits in official 32-class order")
                torch.cuda.synchronize(device)
                if index >= warmup:
                    inference.append(time.perf_counter() - tick)
                del logits
        report.update({
            "status": "ok", "training_update_seconds": durations,
            "training_update_seconds_median": sorted(durations)[len(durations) // 2],
            "training_microbatch_seconds_median": sorted(durations)[len(durations) // 2] / accumulation,
            "inference_batch_seconds": inference,
            "inference_batch_size": 1,
            "inference_seconds_per_image": sorted(inference)[len(inference) // 2],
            "training_peak_allocated_gib": train_peak / 1024**3,
            "training_peak_reserved_gib": reserved_peak / 1024**3,
            "inference_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        })
        report["proposed_pilot_budget"] = proposed_budget(report, hours=pilot_hours)
    except torch.cuda.OutOfMemoryError:
        report.update({"status": "oom", "reason": "CUDA out of memory at this configuration",
                       "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3})
    report["probe_wall_seconds"] = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True, help="New JSON path inside scratch workspace")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--pilot-hours", type=float, default=1.0)
    args = parser.parse_args()
    result = probe(args.config, args.report, device_name=args.device, steps=args.steps,
                   warmup=args.warmup, pilot_hours=args.pilot_hours)
    print(json.dumps(result, indent=2))
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
