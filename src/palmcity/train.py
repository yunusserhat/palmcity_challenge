"""Train full-panorama models with a fixed budget and epoch-boundary resume."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .data import CLASS_NAMES, load_mask, read_manifest
from .metrics import confusion_matrix, scores_from_confusion
from .models import (
    NUM_CLASSES, PREPROCESSING, amp_settings, build_model, image_tensor,
    inference_model_config, optimizer_parameter_groups, select_device,
)
from .storage import require_free_space, require_workspace, safe_output_path
from .training_state import (
    atomic_checkpoint, atomic_json, capture_rng, code_identity, cpu_state,
    data_identity, dependency_versions, json_sha256, restore_rng, write_metrics,
)


class PanoramaDataset(Dataset):
    def __init__(self, records: list[dict[str, str]], config: dict[str, Any], *, training: bool) -> None:
        self.records = records
        self.image_size = tuple(int(value) for value in config["image_size"])
        self.training = training
        augmentation = config.get("augmentation", {})
        self.flip_probability = float(augmentation.get("horizontal_flip", 0.5)) if training else 0.0
        self.roll_probability = float(augmentation.get("horizontal_roll", 0.0)) if training else 0.0
        if len(self.image_size) != 2 or self.image_size[0] <= 0 or self.image_size[1] != 2 * self.image_size[0]:
            raise ValueError("image_size must preserve the full 2:1 panorama")
        if not 0 <= self.flip_probability <= 1 or not 0 <= self.roll_probability <= 1:
            raise ValueError("Augmentation probabilities must be between 0 and 1")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        with Image.open(record["image"]) as image:
            native_size = image.size
            image_data = image_tensor(image, self.image_size)
        target = load_mask(record["mask"])
        if target.shape != (native_size[1], native_size[0]):
            raise ValueError(f"Image and mask dimensions differ for {record['id']}")
        if self.training:
            target = np.asarray(
                Image.fromarray(target).resize((self.image_size[1], self.image_size[0]), Image.Resampling.NEAREST),
                dtype=np.uint8,
            )
        target_data = torch.from_numpy(target.astype(np.int64, copy=True))
        if self.flip_probability and random.random() < self.flip_probability:
            image_data = image_data.flip(-1)
            target_data = target_data.flip(-1)
        if self.roll_probability and random.random() < self.roll_probability:
            shift = random.randrange(self.image_size[1])
            image_data = image_data.roll(shift, dims=-1)
            target_data = target_data.roll(shift, dims=-1)
        return image_data, target_data


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, dice_weight: float) -> torch.Tensor:
    loss = F.cross_entropy(logits.float(), target)
    if dice_weight:
        probabilities = logits.float().softmax(dim=1)
        one_hot = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
        dimensions = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(dim=dimensions)
        denominator = probabilities.sum(dim=dimensions) + one_hot.sum(dim=dimensions)
        dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
        loss = loss + dice_weight * (1 - dice.mean())
    return loss


@torch.inference_mode()
def validate(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool, amp_dtype: torch.dtype) -> dict[str, Any]:
    model.eval()
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(images)
        if logits.ndim != 4 or logits.shape[1] != NUM_CLASSES or not torch.isfinite(logits).all():
            raise RuntimeError("Validation model must produce finite logits for all 32 classes")
        logits = F.interpolate(logits.float(), size=targets.shape[-2:], mode="bilinear", align_corners=False)
        predictions = logits.argmax(dim=1).cpu().numpy()
        matrix += confusion_matrix(targets.numpy(), predictions, NUM_CLASSES)
    return scores_from_confusion(matrix)



def model_training_loss(model: nn.Module, images: torch.Tensor, targets: torch.Tensor, dice_weight: float) -> torch.Tensor:
    """Use the backend's native query loss when required; retain dense CE/Dice."""
    if bool(getattr(model, "uses_native_loss", False)):
        if dice_weight:
            raise ValueError("Native model loss requires top-level dice_weight=0")
        return model.training_loss(images, targets)
    logits = model(images)
    if logits.ndim != 4 or logits.shape[1] != NUM_CLASSES:
        raise RuntimeError("Training model must produce logits for all 32 classes")
    if logits.shape[-2:] != targets.shape[-2:]:
        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
    return segmentation_loss(logits, targets, dice_weight)


def accumulation_samples(batch_index: int, batches: int, samples: int, batch_size: int,
                         accumulation: int, drop_last: bool) -> int:
    """Weight microbatches by images, including a short final batch/window."""
    start = (batch_index // accumulation) * accumulation
    end = min(start + accumulation, batches)
    if drop_last:
        return (end - start) * batch_size
    return min(end * batch_size, samples) - start * batch_size


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _best_checkpoint(latest: dict[str, Any]) -> dict[str, Any]:
    return {key: latest[key] for key in (
        "format_version", "epoch", "best_miou", "class_names", "preprocessing", "config", "state_dict",
        "identity", "optimizer_steps",
    )}


def train(
    config_path: str | Path,
    manifest_path: str | Path,
    run_name: str,
    *,
    device_name: str = "cuda:0",
    epochs_override: int | None = None,
    smoke: bool = False,
    allow_pretrained_downloads: bool = False,
    resume: str | Path | None = None,
    stop_after_epochs: int | None = None,
    max_optimizer_steps_override: int | None = None,
    validation_every_epochs_override: int | None = None,
) -> Path:
    """Resume only latest.pt in the same run, retaining the original schedule.

    Checkpoints are committed after each epoch and its scheduled validation. A process
    failure within an epoch replays that epoch from the last committed state.
    stop_after_epochs is an operational pause, not a change to the planned budget.
    """
    workspace = require_workspace()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name):
        raise ValueError("run_name must be a simple directory name")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Training config must be a JSON object")
    config = copy.deepcopy(config)
    if epochs_override is not None:
        config["epochs"] = epochs_override
    if max_optimizer_steps_override is not None:
        config["max_optimizer_steps"] = max_optimizer_steps_override
    if validation_every_epochs_override is not None:
        config["validation_every_epochs"] = validation_every_epochs_override
    if smoke:
        config.update({"epochs": 1, "batch_size": 1, "num_workers": 0, "image_size": [32, 64], "amp": False,
                       "warmup_steps": 0, "max_optimizer_steps": None, "validation_every_epochs": 1,
                       "gradient_checkpointing": False})
        config.pop("validation_epochs", None)
        config["model"] = {"architecture": "tiny", "classes": NUM_CLASSES, "encoder_weights": None}
    epochs = int(config.get("epochs", 50))
    batch_size = int(config.get("batch_size", 2))
    workers = int(config.get("num_workers", 2))
    accumulation = int(config.get("gradient_accumulation", 1))
    dice_weight = float(config.get("dice_weight", 0.0))
    lr = float(config.get("learning_rate", 1e-4))
    minimum_lr = float(config.get("minimum_learning_rate", 1e-6))
    weight_decay = float(config.get("weight_decay", 1e-2))
    power = float(config.get("polynomial_power", 0.9))
    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    backbone_lr_multiplier = float(config.get("backbone_lr_multiplier", 1.0))
    warmup_steps = int(config.get("warmup_steps", 0))
    schedule_name = str(config.get("scheduler", "polynomial"))
    validation_every = int(config.get("validation_every_epochs", 1))
    validation_epochs = config.get("validation_epochs")
    max_steps = config.get("max_optimizer_steps")
    if max_steps is not None:
        max_steps = int(max_steps)
    if min(epochs, batch_size, accumulation) <= 0 or workers < 0:
        raise ValueError("epochs, batch_size and gradient_accumulation must be positive; workers nonnegative")
    if stop_after_epochs is not None and stop_after_epochs <= 0:
        raise ValueError("stop_after_epochs must be positive")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_optimizer_steps must be positive")
    if schedule_name not in {"polynomial", "cosine", "constant"} or warmup_steps < 0:
        raise ValueError("scheduler must be polynomial/cosine/constant; warmup_steps nonnegative")
    if validation_every <= 0:
        raise ValueError("validation_every_epochs must be positive")
    if validation_epochs is not None:
        if "validation_every_epochs" in config:
            raise ValueError("Choose validation_epochs or validation_every_epochs, not both")
        if not isinstance(validation_epochs, list) or any(type(value) is not int or not 1 <= value <= epochs for value in validation_epochs):
            raise ValueError("validation_epochs must list epoch integers inside the planned training budget")
        if validation_epochs != sorted(set(validation_epochs)):
            raise ValueError("validation_epochs must be sorted and unique")
    if not (math.isfinite(lr) and lr > 0 and math.isfinite(minimum_lr) and 0 <= minimum_lr <= lr):
        raise ValueError("Learning rates must be finite and 0 <= minimum_learning_rate <= learning_rate")
    if not (math.isfinite(dice_weight) and dice_weight >= 0 and math.isfinite(weight_decay) and weight_decay >= 0 and math.isfinite(power) and power > 0):
        raise ValueError("Loss weight, weight decay and schedule power must be finite and valid")
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    if not math.isfinite(backbone_lr_multiplier) or not 0 < backbone_lr_multiplier <= 1:
        raise ValueError("backbone_lr_multiplier must be finite and in (0,1]")
    manifest = read_manifest(manifest_path)
    train_records, val_records = manifest.get("train", []), manifest.get("val", [])
    if not train_records or not val_records:
        raise ValueError("Training requires nonempty train and val manifest splits")
    if smoke:
        train_records, val_records = train_records[:2], val_records[:1]
    architecture = str(config["model"].get("architecture", "deeplabv3plus")).lower()
    pooled_batchnorm = architecture in {"deeplabv3plus", "upernet"}
    drop_last = bool(config.get("drop_last", pooled_batchnorm and batch_size > 1))
    if pooled_batchnorm and (batch_size < 2 or len(train_records) < 2):
        raise ValueError(f"{architecture} training needs at least two images per batch for its pooled BatchNorm")
    if pooled_batchnorm and not drop_last and len(train_records) % batch_size == 1:
        raise ValueError(f"Set drop_last=true to avoid a singleton {architecture} BatchNorm batch")
    if drop_last and len(train_records) < batch_size:
        raise ValueError("drop_last would remove every training batch; reduce batch_size")
    run_dir = safe_output_path(Path("runs") / run_name, workspace)
    latest_path, checkpoint_path = run_dir / "latest.pt", run_dir / "best.pt"
    if resume is None and run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    if resume is not None and safe_output_path(resume, workspace) != latest_path:
        raise ValueError("Resume requires latest.pt in the same named run directory")
    require_free_space(workspace, gib=0.02 if smoke else 1.0)
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    deterministic = bool(config.get("deterministic", False))
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = bool(config.get("cudnn_benchmark", False)) and not deterministic
    device = select_device(device_name)
    use_amp, amp_dtype = amp_settings(device, bool(config.get("amp", True)))
    config.update({"epochs": epochs, "drop_last": drop_last, "class_names": list(CLASS_NAMES),
                   "preprocessing": PREPROCESSING, "smoke": smoke})
    training = PanoramaDataset(train_records, config, training=True)
    validation = PanoramaDataset(val_records, config, training=False)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        training, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=device.type == "cuda", worker_init_fn=seed_worker, generator=generator, drop_last=drop_last,
    )
    val_loader = DataLoader(validation, batch_size=1, shuffle=False, num_workers=workers,
                            pin_memory=device.type == "cuda", worker_init_fn=seed_worker)
    planned_updates = epochs * math.ceil(len(train_loader) / accumulation)
    total_updates = min(max_steps, planned_updates) if max_steps is not None else planned_updates
    if max_steps is not None and max_steps > planned_updates:
        raise ValueError("epochs do not provide enough batches for max_optimizer_steps")
    if warmup_steps >= total_updates:
        raise ValueError("warmup_steps must be smaller than the planned optimizer budget")
    identity = {
        "config_sha256": json_sha256(config), "data": data_identity(manifest, manifest_path),
        "code": code_identity(), "dependencies": dependency_versions(), "seed": seed,
        "device": str(device), "amp_enabled": use_amp, "amp_dtype": str(amp_dtype),
        "optimizer_budget": total_updates, "resume_boundary": "after_epoch",
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        identity["gpu"] = {"name": properties.name, "total_memory_bytes": properties.total_memory,
                           "capability": [properties.major, properties.minor]}
    saved: dict[str, Any] | None = None
    if resume is not None:
        saved = torch.load(latest_path, map_location="cpu", weights_only=True)
        if saved.get("training_format_version") != 1 or saved.get("identity") != identity:
            raise ValueError("Resume identity differs: config, data, code, dependencies, seed, device or schedule changed")
    # On resume no pretrained download is needed: construct random weights, then restore.
    model_config = inference_model_config(config["model"]) if saved is not None else config["model"]
    model = build_model(model_config, allow_pretrained_downloads=allow_pretrained_downloads).to(device)
    if bool(getattr(model, "uses_native_loss", False)) and dice_weight:
        raise ValueError("Native model loss requires top-level dice_weight=0")
    if bool(config.get("gradient_checkpointing", False)):
        if not hasattr(model, "gradient_checkpointing_enable"):
            raise ValueError("This backend does not expose gradient_checkpointing_enable")
        model.gradient_checkpointing_enable()
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    # best weights + latest model/Adam state + atomic temporary; generous tensor/serialization margin.
    checkpoint_peak_bytes = 10 * parameter_bytes
    require_free_space(workspace, gib=max(0.02, checkpoint_peak_bytes / 1024**3))
    parameter_groups = optimizer_parameter_groups(model, lr, weight_decay, backbone_lr_multiplier)
    if not parameter_groups:
        raise ValueError("The model has no trainable parameters")
    optimizer = torch.optim.AdamW(parameter_groups, lr=lr, weight_decay=weight_decay)

    def lr_multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, max(0.0, (step - warmup_steps) / (total_updates - warmup_steps)))
        if schedule_name == "constant":
            return 1.0
        decay = (1 - progress) ** power if schedule_name == "polynomial" else (1 + math.cos(math.pi * progress)) / 2
        return max(minimum_lr / lr, decay)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    history: list[dict[str, Any]] = []
    best_score, best_epoch, first_epoch, optimizer_steps = -1.0, 0, 1, 0
    elapsed_before = training_seconds = validation_seconds = 0.0
    peak_allocated = peak_reserved = 0
    if saved is not None:
        model.load_state_dict(saved["state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        restore_rng(saved["rng"], generator, device)
        best_score, best_epoch = float(saved["best_miou"]), int(saved["best_epoch"])
        first_epoch, optimizer_steps = int(saved["epoch"]) + 1, int(saved["optimizer_steps"])
        history = saved["metrics_history"]
        elapsed_before = float(saved["elapsed_seconds"])
        training_seconds, validation_seconds = saved["training_seconds"], saved["validation_seconds"]
        peak_allocated, peak_reserved = saved["peak_allocated_bytes"], saved["peak_reserved_bytes"]
        # Repair a failure between latest commit and the derived best/log writes.
        if best_epoch == int(saved["epoch"]):
            atomic_checkpoint(checkpoint_path, _best_checkpoint(saved))
        elif best_epoch > 0:
            existing_best = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if int(existing_best["epoch"]) != best_epoch:
                raise ValueError("Best checkpoint does not match the committed training state")
        write_metrics(run_dir / "metrics.jsonl", history)
    else:
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(exist_ok=False)
        atomic_json(run_dir / "config.json", config)
        atomic_json(run_dir / "metadata.json", {**identity, "parameter_count": sum(p.numel() for p in model.parameters()),
                                                "checkpoint_peak_estimate_bytes": checkpoint_peak_bytes,
                                                "optimizer_groups": [{"name": group["group_name"], "learning_rate": group["initial_lr"],
                                                                      "weight_decay": group["weight_decay"],
                                                                      "parameter_count": sum(p.numel() for p in group["params"])}
                                                                     for group in optimizer.param_groups],
                                                "model_initialization": "pretrained" if any(config["model"].get(key) for key in ("encoder_weights", "encoder_pretrained_path", "pretrained_model_name_or_path", "pretrained_backbone_name_or_path")) else "random"})
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    started = time.monotonic()
    epochs_this_call = 0
    for epoch in range(first_epoch, epochs + 1):
        if optimizer_steps >= total_updates:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, sample_count = 0.0, 0
        epoch_batches = min(len(train_loader), (total_updates - optimizer_steps) * accumulation)
        _synchronize(device)
        train_started = time.monotonic()
        for batch_index, (images, targets) in enumerate(train_loader):
            if batch_index >= epoch_batches:
                break
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            group_samples = accumulation_samples(batch_index, epoch_batches, len(training), batch_size, accumulation, drop_last)
            if hasattr(model, "set_training_progress"):
                model.set_training_progress(optimizer_steps, total_updates)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = model_training_loss(model, images, targets, dice_weight)
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite or non-scalar training loss in epoch {epoch}")
            scaler.scale(loss * images.shape[0] / group_samples).backward()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == epoch_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    optimizer_steps += 1
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach()) * images.shape[0]
            sample_count += images.shape[0]
        _synchronize(device)
        epoch_training_seconds = time.monotonic() - train_started
        training_seconds += epoch_training_seconds
        scheduled_validation = epoch in validation_epochs if validation_epochs is not None else epoch % validation_every == 0
        should_validate = scheduled_validation or epoch == epochs or optimizer_steps >= total_updates
        scores = None
        epoch_validation_seconds = 0.0
        if should_validate:
            validation_started = time.monotonic()
            scores = validate(model, val_loader, device, use_amp, amp_dtype)
            _synchronize(device)
            epoch_validation_seconds = time.monotonic() - validation_started
        validation_seconds += epoch_validation_seconds
        if device.type == "cuda":
            peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))
        score = float(scores["miou"]) if scores is not None else None
        if score is not None and not math.isfinite(score):
            raise RuntimeError("Validation produced a nonfinite mIoU")
        improved = score is not None and score > best_score
        if improved:
            best_score, best_epoch = score, epoch
        metrics = {"epoch": epoch, "epoch_complete": epoch_batches == len(train_loader),
                   "optimizer_steps": optimizer_steps, "train_samples": sample_count,
                   "train_loss": loss_sum / sample_count, "validation_performed": should_validate,
                   "val_miou": score, "val_mf1": float(scores["mf1"]) if scores is not None else None,
                   "val_class_iou": scores["iou"] if scores is not None else None,
                   "val_class_f1": scores["f1"] if scores is not None else None,
                   "learning_rate": optimizer.param_groups[0]["lr"],
                   "learning_rates": {group["group_name"]: group["lr"] for group in optimizer.param_groups},
                   "training_seconds": epoch_training_seconds, "validation_seconds": epoch_validation_seconds,
                   "elapsed_seconds": elapsed_before + time.monotonic() - started,
                   "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved}
        history.append(metrics)
        latest = {
            "format_version": 1, "training_format_version": 1, "epoch": epoch,
            "best_miou": best_score, "best_epoch": best_epoch, "optimizer_steps": optimizer_steps,
            "class_names": list(CLASS_NAMES), "preprocessing": PREPROCESSING, "config": config,
            "state_dict": cpu_state(model.state_dict()), "optimizer": cpu_state(optimizer.state_dict()),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "rng": capture_rng(generator, device),
            "identity": identity, "metrics_history": history, "elapsed_seconds": metrics["elapsed_seconds"],
            "training_seconds": training_seconds, "validation_seconds": validation_seconds,
            "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
        }
        # Commit authoritative state first, then recoverable inference checkpoint and metrics.
        atomic_checkpoint(latest_path, latest)
        if improved:
            atomic_checkpoint(checkpoint_path, _best_checkpoint(latest))
        write_metrics(run_dir / "metrics.jsonl", history)
        print(json.dumps(metrics, allow_nan=False), flush=True)
        epochs_this_call += 1
        if stop_after_epochs is not None and epochs_this_call >= stop_after_epochs:
            break
    atomic_json(run_dir / "summary.json", {
        "best_miou": best_score if best_epoch else None, "best_epoch": best_epoch, "optimizer_steps": optimizer_steps,
        "planned_optimizer_steps": total_updates, "epochs_committed": len(history),
        "validation_evaluations": sum(row["validation_performed"] for row in history),
        "status": "complete" if optimizer_steps >= total_updates or len(history) >= epochs else "paused",
        "training_seconds": training_seconds, "validation_seconds": validation_seconds,
        "elapsed_seconds": elapsed_before + time.monotonic() - started,
        "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
        "synthetic_smoke": smoke, "best_checkpoint": str(checkpoint_path), "resume_checkpoint": str(latest_path),
    })
    return checkpoint_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-name", required=True, help="A new run name, or the same name when resuming latest.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-optimizer-steps", type=int, help="A fixed pilot update cap; record separately from measured compute time")
    parser.add_argument("--validation-every-epochs", type=int, help="Reserve validation opportunities at this interval and always at budget end")
    parser.add_argument("--stop-after-epochs", type=int, help="Pause after this many additional epochs, preserving the schedule")
    parser.add_argument("--resume", help="Latest checkpoint in the same named run; config/data/code/dependencies must match")
    parser.add_argument("--smoke", action="store_true", help="Use the tiny download-free model and at most two training images")
    parser.add_argument("--allow-pretrained-downloads", action="store_true", help="Explicitly allow pretrained initialization")
    args = parser.parse_args(argv)
    checkpoint = train(args.config, args.manifest, args.run_name, device_name=args.device,
                       epochs_override=args.epochs, smoke=args.smoke,
                       allow_pretrained_downloads=args.allow_pretrained_downloads,
                       resume=args.resume, stop_after_epochs=args.stop_after_epochs,
                       max_optimizer_steps_override=args.max_optimizer_steps,
                       validation_every_epochs_override=args.validation_every_epochs)
    if checkpoint.exists():
        print(f"Best checkpoint: {checkpoint}")
    else:
        print(f"Paused before the first scheduled validation; resume: {checkpoint.parent / 'latest.pt'}")


if __name__ == "__main__":
    main()
