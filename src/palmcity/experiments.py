"""Calibrate real loaders, create bounded budgets, execute and summarize runs."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
import random
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .benchmark import idle_gpu
from .data import CLASS_NAMES, read_manifest
from .models import NUM_CLASSES, amp_settings, build_model, optimizer_parameter_groups, select_device
from .storage import require_free_space, require_workspace, safe_output_path
from .train import PanoramaDataset, model_training_loss, seed_worker, validate
from .training_state import atomic_json, code_identity, data_identity, dependency_versions, file_sha256

PRETRAINED_FIELDS = ("encoder_weights", "encoder_pretrained_path", "pretrained_model_name_or_path", "pretrained_backbone_name_or_path")


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _initialization(config: dict[str, Any]) -> str:
    return "pretrained" if any(config["model"].get(key) for key in PRETRAINED_FIELDS) else "random"


def _updates_per_epoch(config: dict[str, Any], count: int) -> int:
    batch, accumulation = int(config["batch_size"]), int(config.get("gradient_accumulation", 1))
    if min(batch, accumulation) <= 0:
        raise ValueError("A positive batch and accumulation are required")
    batches = count // batch if config.get("drop_last", False) else math.ceil(count / batch)
    if min(batch, accumulation, batches) <= 0:
        raise ValueError("A positive batch, accumulation and nonempty training loader are required")
    return math.ceil(batches / accumulation)


def calibrate(config_path: str | Path, manifest_path: str | Path, report_path: str | Path, *,
              device_name: str = "cuda:0", warmup: int = 1, steps: int = 3, val_images: int = 3,
              allow_pretrained_downloads: bool = False) -> dict[str, Any]:
    """Time a bounded real-data training probe; discard its updated parameters.

    No metric is selected from this probe. Validation images only time inference,
    and no test image or hidden label enters the optimizer or timing samples.
    Pretrained construction is cache-only even with the explicit initialization flag.
    """
    if not 1 <= warmup <= 3 or not 1 <= steps <= 5 or not 1 <= val_images <= 10:
        raise ValueError("Calibration needs 1..3 warmups, 1..5 measured updates and 1..10 val images")
    workspace = require_workspace()
    output = safe_output_path(report_path, workspace)
    if output.exists():
        raise FileExistsError(output)
    config = copy.deepcopy(_read_json(config_path))
    manifest = read_manifest(manifest_path)
    if not manifest["train"] or not manifest["val"]:
        raise ValueError("Calibration requires nonempty train and public val splits")
    batch, accumulation = int(config["batch_size"]), int(config.get("gradient_accumulation", 1))
    if min(batch, accumulation) <= 0:
        raise ValueError("Calibration batch and gradient accumulation must be positive")
    architecture = str(config["model"].get("architecture", ""))
    config.setdefault("drop_last", architecture in {"deeplabv3plus", "upernet"} and batch > 1)
    if architecture in {"deeplabv3plus", "upernet"} and (batch < 2 or (not config["drop_last"] and len(manifest["train"]) % batch == 1)):
        raise ValueError("Pooled BatchNorm needs at least two images in every training batch")
    device_spec = torch.device(device_name)
    if device_spec.type == "cuda" and device_spec.index is None:
        raise ValueError("Use an explicit physical cuda:N")
    gpu = idle_gpu(device_spec.index) if device_spec.type == "cuda" else None
    device = select_device(device_name)
    # Acquisition is a separate, explicitly authorized operation; calibration cannot download.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    require_free_space(workspace, gib=1.0)
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    deterministic = bool(config.get("deterministic", False))
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = bool(config.get("cudnn_benchmark", False)) and not deterministic
    use_amp, dtype = amp_settings(device, bool(config.get("amp", True)))
    data = data_identity(manifest, manifest_path)
    report: dict[str, Any] = {
        "schema_version": 1, "kind": "real_loader_calibration", "status": "initializing",
        "quality_metrics": None, "candidate_name": Path(config_path).stem,
        "config_path": str(Path(config_path).resolve()), "config_sha256": file_sha256(config_path),
        "model_config": config["model"], "initialization": _initialization(config),
        "manifest_path": str(Path(manifest_path).resolve()), "data": data,
        "code": code_identity(), "dependencies": dependency_versions(), "gpu": gpu,
        "device": str(device), "image_size": config["image_size"], "batch_size": batch,
        "gradient_accumulation": accumulation, "drop_last": config["drop_last"],
        "seed": seed, "amp": use_amp, "amp_dtype": str(dtype),
        "warmup_updates": warmup, "measured_updates": steps,
        "includes": ["real image/mask decoding", "configured augmentation", "DataLoader", "host to device", "native loss", "backward", "gradient clipping", "AdamW"],
        "excludes": ["model initialization", "checkpoint IO"],
    }
    started = time.perf_counter()
    try:
        model = build_model(config["model"], allow_pretrained_downloads=allow_pretrained_downloads).to(device)
        if config.get("gradient_checkpointing"):
            enable = getattr(model, "gradient_checkpointing_enable", None)
            if enable is None:
                raise ValueError("Backend does not support requested gradient checkpointing")
            enable()
        parameters = sum(p.numel() for p in model.parameters())
        report["parameter_count"] = parameters
        report["checkpoint_peak_estimate_bytes"] = parameters * 4 * 10
        require_free_space(workspace, gib=max(1.0, report["checkpoint_peak_estimate_bytes"] / 1024**3))
        dataset = PanoramaDataset(manifest["train"], config, training=True)
        loader = DataLoader(dataset, batch_size=batch, shuffle=True,
                            num_workers=int(config.get("num_workers", 2)), drop_last=config["drop_last"],
                            pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
                            generator=torch.Generator().manual_seed(seed))
        if len(loader) == 0:
            raise ValueError("Calibration training loader is empty")
        iterator = iter(loader)
        lr = float(config["learning_rate"])
        groups = optimizer_parameter_groups(model, lr, float(config.get("weight_decay", 0.01)),
                                            float(config.get("backbone_lr_multiplier", 1.0)))
        optimizer = torch.optim.AdamW(groups, lr=lr)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype == torch.float16)
        model.train()
        durations, observed_samples, losses = [], [], []
        completed = 0
        for update in range(warmup + steps):
            if update == warmup and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=True)
            _synchronize(device)
            tick = time.perf_counter()
            microbatches = []
            for _ in range(accumulation):
                try:
                    microbatches.append(next(iterator))
                except StopIteration:
                    iterator = iter(loader)
                    microbatches.append(next(iterator))
            group_samples = sum(images.shape[0] for images, _ in microbatches)
            loss_sum = 0.0
            for images, targets in microbatches:
                images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                if hasattr(model, "set_training_progress"):
                    model.set_training_progress(0, 1)
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                    loss = model_training_loss(model, images, targets, float(config.get("dice_weight", 0.0)))
                if loss.ndim != 0 or not torch.isfinite(loss):
                    raise RuntimeError("Calibration produced nonfinite or non-scalar native loss")
                scaler.scale(loss * images.shape[0] / group_samples).backward()
                loss_sum += float(loss.detach()) * images.shape[0]
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < before:
                raise RuntimeError("Calibration overflow skipped an optimizer update; revise AMP recipe")
            completed += 1
            _synchronize(device)
            if update >= warmup:
                durations.append(time.perf_counter() - tick)
                observed_samples.append(group_samples)
                losses.append(loss_sum / group_samples)
        train_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        reserved_peak = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        del optimizer, scaler, iterator, loader, microbatches, images, targets, loss
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        validation = PanoramaDataset(manifest["val"][:val_images], config, training=False)
        inference_loader = DataLoader(validation, batch_size=1, shuffle=False, num_workers=0)
        inference_warmup = DataLoader(PanoramaDataset(manifest["val"][:1], config, training=False),
                                     batch_size=1, shuffle=False, num_workers=0)
        validate(model, inference_warmup, device, use_amp, dtype)
        _synchronize(device)
        validation_started = time.perf_counter()
        # Time the same native-resolution GT/confusion work as the trainer; discard
        # these tiny-subset scores so the timing probe cannot become a quality trial.
        validate(model, inference_loader, device, use_amp, dtype)
        _synchronize(device)
        validation_probe_seconds = time.perf_counter() - validation_started
        report.update({"status": "ok", "training_update_seconds": durations,
                       "training_update_seconds_median": statistics.median(durations),
                       "training_update_seconds_max": max(durations), "training_samples_per_update": observed_samples,
                       "calibration_train_loss": losses, "optimizer_updates_executed": completed,
                       "inference_seconds_per_image": validation_probe_seconds / len(validation),
                       "validation_probe_seconds": validation_probe_seconds,
                       "validation_probe_images": len(validation),
                       "inference_warmup_images": 1,
                       "validation_timing_includes": ["image/mask decoding", "host to device", "model", "native-resolution logit alignment", "32-class confusion matrix"],
                       "estimated_updates_per_epoch": _updates_per_epoch(config, len(manifest["train"])),
                       "training_peak_allocated_bytes": train_peak, "training_peak_reserved_bytes": reserved_peak,
                       "inference_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0})
    except (RuntimeError, ValueError, OSError, ImportError) as error:
        report.update({"status": "oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "failed",
                       "error_type": type(error).__name__, "reason": str(error)[:3000]})
    report["calibration_wall_seconds"] = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    return report


def _uniform_validation_epochs(epochs: int, evaluations: int) -> list[int]:
    if epochs < evaluations:
        raise ValueError("The compute budget cannot support the reserved validation evaluations")
    return [math.ceil(index * epochs / evaluations) for index in range(1, evaluations + 1)]


def budget_from_calibrations(reports: list[dict[str, Any]], *, seconds: float | None = None,
                             evaluations: int = 10, phase: str = "pilot", seeds: list[int] | None = None) -> dict[str, Any]:
    """Hold real training compute and reserved validation counts approximately equal."""
    if not reports or not 2 <= evaluations <= 10:
        raise ValueError("Supply calibration reports and 2..10 reserved validation evaluations")
    successes = [report for report in reports if report.get("status") == "ok"]
    if not successes:
        raise ValueError("At least one successful real-loader calibration is required")
    if any(report.get("kind") != "real_loader_calibration" or report.get("schema_version") != 1 for report in reports):
        raise ValueError("Use version-1 real-loader calibration reports")
    if phase not in {"pilot", "confirmation"}:
        raise ValueError("phase must be pilot or confirmation")
    if phase == "confirmation" and len(successes) > 2:
        raise ValueError("Select the best two declared candidates before longer confirmation runs")
    seeds = [42] if seeds is None and phase == "pilot" else [42, 123, 2026] if seeds is None else seeds
    if not seeds or len(seeds) != len(set(seeds)) or any(type(seed) is not int for seed in seeds):
        raise ValueError("Seeds must be distinct integers")
    if phase == "pilot" and len(seeds) != 1:
        raise ValueError("Initial pilots use one declared seed; use confirmation for multiple seeds")
    conditions = {(tuple(report["image_size"]), report["batch_size"] * report["gradient_accumulation"],
                   report["amp_dtype"], report["data"]["manifest_sha256"], report["data"]["input_sha256"])
                  for report in successes}
    if len(conditions) != 1:
        raise ValueError("Calibrations must share resolution, effective batch, AMP dtype and input content")
    names = [report["candidate_name"] for report in reports]
    if len(names) != len(set(names)):
        raise ValueError("Candidate calibration names must be unique")
    for report in successes:
        durations = (report["training_update_seconds_median"], report["inference_seconds_per_image"])
        if any(not math.isfinite(value) or value <= 0 for value in durations):
            raise ValueError("Calibration timing values must be finite and positive")
    required = max(evaluations * report["estimated_updates_per_epoch"] * report["training_update_seconds_median"]
                   + evaluations * report["data"]["counts"]["val"] * report["inference_seconds_per_image"]
                   for report in successes)
    if seconds is None:
        seconds = max(300.0, min(1800.0, required))
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Target seconds must be finite and positive")
    if required > seconds + 1e-9:
        raise ValueError(f"Need at least {required:.1f}s per candidate for {evaluations} reserved validations; raise the declared budget")
    rows = []
    for report in successes:
        validation_seconds = evaluations * report["data"]["counts"]["val"] * report["inference_seconds_per_image"]
        updates = max(1, int((seconds - validation_seconds) / report["training_update_seconds_median"]))
        # Roundoff at the exact minimum must still leave the declared evaluation count.
        updates = max(updates, evaluations * report["estimated_updates_per_epoch"])
        epochs = math.ceil(updates / report["estimated_updates_per_epoch"])
        recipe = {"epochs": epochs, "max_optimizer_steps": updates,
                  "validation_epochs": _uniform_validation_epochs(epochs, evaluations),
                  "warmup_steps": min(updates - 1, max(1, int(0.05 * updates)))}
        for seed in seeds:
            rows.append({"candidate_name": report["candidate_name"], "cohort": report["initialization"],
                         "phase": phase, "seed": seed, "source_config_path": report["config_path"],
                         "source_config_sha256": report["config_sha256"], "recipe": {**recipe, "seed": seed},
                         "target_train_and_validation_seconds": seconds,
                         "estimated_train_seconds": updates * report["training_update_seconds_median"],
                         "estimated_validation_seconds": validation_seconds,
                         "calibration": report})
    first = successes[0]
    return {"schema_version": 1, "kind": "bounded_experiment_plan", "phase": phase,
            "manifest_path": first["manifest_path"], "data": first["data"],
            "target_seconds_per_run": seconds, "reserved_validation_evaluations": evaluations,
            "seeds": seeds, "runs": rows,
            "unavailable_candidates": [{"candidate_name": report["candidate_name"], "cohort": report["initialization"],
                                        "status": report["status"], "reason": report.get("reason")}
                                       for report in reports if report.get("status") != "ok"],
            "budget_basis": "real-loader AdamW updates plus reserved serial validation inference estimate",
            "budget_excludes": ["startup", "checkpoint IO"],
            "selection_policy": "Separate initialization cohorts; pilot rankings need longer multi-seed confirmation"}


def write_plan(report_paths: list[str | Path], output_dir: str | Path, *, seconds: float | None = None,
               evaluations: int = 10, phase: str = "pilot", seeds: list[int] | None = None) -> Path:
    workspace = require_workspace()
    directory = safe_output_path(output_dir, workspace)
    if directory.exists():
        raise FileExistsError(directory)
    reports = [_read_json(path) for path in report_paths]
    plan = budget_from_calibrations(reports, seconds=seconds, evaluations=evaluations, phase=phase, seeds=seeds)
    plan["code"] = code_identity()
    plan["dependencies"] = dependency_versions()
    directory.mkdir(parents=True, exist_ok=False)
    for row in plan["runs"]:
        source = Path(row["source_config_path"])
        if file_sha256(source) != row["source_config_sha256"]:
            raise ValueError(f"Source config changed after calibration: {source}")
        config = _read_json(source)
        config.update(row["recipe"])
        config.pop("validation_every_epochs", None)
        name = f'{phase}-{row["candidate_name"]}-seed{row["seed"]}'
        if not all(character.isalnum() or character in "_.-" for character in name):
            raise ValueError("Candidate names must form simple run directory names")
        config_path = directory / f"{name}.json"
        atomic_json(config_path, config)
        row.update({"run_name": name, "config_path": str(config_path), "config_sha256": file_sha256(config_path)})
    path = directory / "plan.json"
    atomic_json(path, plan)
    return path


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run_plan(plan_path: str | Path, *, device_name: str = "cuda:0", names: list[str] | None = None,
             fail_fast: bool = False) -> dict[str, Any]:
    """Run a fixed queue sequentially, recording durable logs and resuming latest.

    Launch this command under the user's normal durable job/session supervisor.
    It does not alter the planned budget or terminate unrelated processes.
    """
    workspace = require_workspace()
    plan = _read_json(plan_path)
    if plan.get("kind") != "bounded_experiment_plan" or plan.get("schema_version") != 1:
        raise ValueError("Use a generated version-1 experiment plan")
    if plan["code"]["sha256"] != code_identity()["sha256"] or plan["dependencies"] != dependency_versions():
        raise ValueError("Code/dependencies changed since planning; regenerate the plan before launching")
    if file_sha256(plan["manifest_path"]) != plan["data"]["manifest_sha256"]:
        raise ValueError("Manifest changed since calibration/planning")
    selected = [row for row in plan["runs"] if names is None or row["run_name"] in names]
    if not selected or names is not None and set(names) != {row["run_name"] for row in selected}:
        raise ValueError("Requested run names must exist in the plan")
    execution = safe_output_path(Path(plan_path).resolve().parent / "execution", workspace)
    execution.mkdir(exist_ok=True)
    device_tag = device_name.replace(":", "-")
    lock = (execution / f"runner-{device_tag}.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise RuntimeError("An experiment runner already owns this plan/device queue") from error
    state: dict[str, Any] = {"plan": str(Path(plan_path).resolve()), "plan_sha256": file_sha256(plan_path),
                             "device": device_name, "pid": os.getpid(), "status": "running", "runs": []}
    state_path = execution / f"state-{device_tag}.json"
    code_root = Path(__file__).resolve().parents[2]
    try:
        atomic_json(state_path, state)
        for row in selected:
            config_path = Path(row["config_path"])
            if file_sha256(config_path) != row["config_sha256"]:
                raise ValueError(f"Planned config changed: {config_path}")
            result_path = execution / f'{row["run_name"]}.json'
            previous = _read_json(result_path) if result_path.exists() else None
            if previous and previous.get("status") == "running" and _pid_alive(previous["pid"]):
                raise RuntimeError(f'Run {row["run_name"]} already has a live process')
            run_dir = safe_output_path(Path("runs") / row["run_name"], workspace)
            summary_path, latest = run_dir / "summary.json", run_dir / "latest.pt"
            if summary_path.exists() and _read_json(summary_path).get("status") == "complete":
                metadata = _read_json(run_dir / "metadata.json")
                if metadata["data"]["input_sha256"] != plan["data"]["input_sha256"]:
                    raise ValueError("Completed run input content differs from this plan")
                actual = _read_json(run_dir / "config.json")
                expected = _read_json(config_path)
                if any(actual.get(key) != value for key, value in expected.items()):
                    raise ValueError("Completed run config differs from this plan")
                result = {"run_name": row["run_name"], "status": "already_complete", "run_dir": str(run_dir)}
                state["runs"].append(result)
                atomic_json(state_path, state)
                continue
            if device_name.startswith("cuda:"):
                idle_gpu(int(device_name.split(":")[1]))
            command = ["bash", str(code_root / "scripts/run.sh"), "python", "-m", "palmcity.train",
                       "--config", str(config_path), "--manifest", plan["manifest_path"],
                       "--run-name", row["run_name"], "--device", device_name]
            if row["cohort"] == "pretrained":
                command.append("--allow-pretrained-downloads")
            if latest.exists():
                if previous and previous.get("device") != device_name:
                    raise ValueError("Resume must use the original device stored in the training identity")
                command.extend(["--resume", str(latest)])
            elif run_dir.exists():
                result = {"run_name": row["run_name"], "status": "failed", "run_dir": str(run_dir),
                          "reason": "Run directory exists without a committed latest.pt; inspect it before starting a new run"}
                atomic_json(result_path, result)
                state["runs"].append(result)
                atomic_json(state_path, state)
                if fail_fast:
                    break
                continue
            log_path = execution / f'{row["run_name"]}.log'
            started = time.time()
            with log_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps({"event": "launch", "utc_timestamp": started, "command": command}) + "\n")
                log.flush()
                child_env = os.environ.copy()
                child_env["HF_HUB_OFFLINE"] = child_env["HF_DATASETS_OFFLINE"] = "1"
                process = subprocess.Popen(command, cwd=code_root, stdout=log, stderr=subprocess.STDOUT,
                                           env=child_env, start_new_session=True)
                result = {"run_name": row["run_name"], "status": "running", "pid": process.pid,
                          "device": device_name, "run_dir": str(run_dir), "log_path": str(log_path),
                          "started_unix_seconds": started, "command": command}
                atomic_json(result_path, result)
                state["current"] = result
                atomic_json(state_path, state)
                print(json.dumps(result), flush=True)
                return_code = process.wait()
            summary = _read_json(summary_path) if summary_path.exists() else None
            result.update({"status": "complete" if return_code == 0 and summary and summary.get("status") == "complete" else "failed",
                           "return_code": return_code, "wall_seconds": time.time() - started,
                           "finished_unix_seconds": time.time(), "summary": summary})
            if result["status"] == "failed":
                result["log_tail"] = log_path.read_text(encoding="utf-8")[-4000:]
            atomic_json(result_path, result)
            state["runs"].append(result)
            state.pop("current", None)
            atomic_json(state_path, state)
            print(json.dumps({key: result[key] for key in ("run_name", "status", "return_code", "wall_seconds")}), flush=True)
            if fail_fast and result["status"] != "complete":
                break
        state["status"] = "complete" if all(row["status"] in {"complete", "already_complete"} for row in state["runs"]) and len(state["runs"]) == len(selected) else "finished_with_failures"
        atomic_json(state_path, state)
        return state
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def aggregate(plan_path: str | Path, *, audit_path: str | Path | None = None) -> dict[str, Any]:
    """Summarize only native public validation metrics; never rank hidden test labels."""
    workspace = require_workspace()
    plan = _read_json(plan_path)
    if plan.get("kind") != "bounded_experiment_plan":
        raise ValueError("Use a bounded experiment plan")
    rare: list[int] = []
    if audit_path is not None:
        audit = _read_json(audit_path)
        if not audit.get("valid") or audit.get("class_names") != CLASS_NAMES or audit.get("input_sha256") != plan["data"]["input_sha256"] or audit.get("manifest_sha256") != plan["data"]["manifest_sha256"]:
            raise ValueError("Audit must validate the same training/validation input content and official class order")
        histogram = np.asarray(audit["class_histograms"]["train"])
        if histogram.shape != (NUM_CLASSES,) or not np.issubdtype(histogram.dtype, np.integer) or np.any(histogram < 0):
            raise ValueError("Audit requires a nonnegative 32-class training histogram")
        rare = sorted(np.flatnonzero(histogram > 0).tolist(), key=lambda i: (int(histogram[i]), i))[:8]
    rows = []
    for run in plan["runs"]:
        directory = safe_output_path(Path("runs") / run["run_name"], workspace)
        row = {key: run[key] for key in ("run_name", "candidate_name", "cohort", "phase", "seed")}
        if not (directory / "summary.json").exists():
            execution = Path(plan_path).resolve().parent / "execution" / f'{run["run_name"]}.json'
            row.update({"status": "not_completed", "execution": _read_json(execution) if execution.exists() else None})
            rows.append(row)
            continue
        summary, metadata, config = (_read_json(directory / name) for name in ("summary.json", "metadata.json", "config.json"))
        expected = _read_json(run["config_path"])
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError(f'Run config differs from plan: {run["run_name"]}')
        if metadata["data"]["input_sha256"] != plan["data"]["input_sha256"] or metadata["seed"] != run["seed"]:
            raise ValueError("Training metadata differs from planned input content or seed")
        if metadata["model_initialization"] != run["cohort"]:
            raise ValueError("Training initialization differs from its declared cohort")
        if config.get("class_names") != CLASS_NAMES or summary.get("synthetic_smoke"):
            raise ValueError("Real experiment aggregation needs all 32 official classes and excludes smoke runs")
        history = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
        validations = [record for record in history if record.get("validation_performed")]
        if summary["validation_evaluations"] != len(validations):
            raise ValueError("Summary and log disagree on reserved validation evaluations")
        row.update({"status": summary["status"], "run_dir": str(directory), "summary": summary,
                    "model_config": config["model"], "validation_evaluations": len(validations),
                    "peak_vram_gib": summary["peak_allocated_bytes"] / 1024**3,
                    "training_seconds": summary["training_seconds"], "validation_seconds": summary["validation_seconds"]})
        if summary.get("best_epoch", 0):
            best = next(record for record in validations if record["epoch"] == summary["best_epoch"])
            iou, f1 = np.asarray(best["val_class_iou"]), np.asarray(best["val_class_f1"])
            if iou.shape != (NUM_CLASSES,) or f1.shape != (NUM_CLASSES,) or not np.isfinite(iou).all() or not np.isfinite(f1).all() or np.any((iou < 0) | (iou > 1)) or np.any((f1 < 0) | (f1 > 1)):
                raise ValueError("Per-class metrics must cover all 32 official classes with finite fractions")
            if not math.isclose(float(iou.mean()), best["val_miou"], rel_tol=0, abs_tol=1e-12) or not math.isclose(float(f1.mean()), best["val_mf1"], rel_tol=0, abs_tol=1e-12):
                raise ValueError("Native validation macros must include all 32 classes including absent classes")
            if not math.isclose(summary["best_miou"], best["val_miou"], rel_tol=0, abs_tol=1e-12) or not math.isclose(summary["best_miou"], max(record["val_miou"] for record in validations), rel_tol=0, abs_tol=1e-12):
                raise ValueError("Best summary score disagrees with validation history")
            row.update({"miou": best["val_miou"], "mf1": best["val_mf1"], "class_iou": iou.tolist(),
                        "class_f1": f1.tolist(), "rare_class_miou": float(iou[rare].mean()) if rare else None,
                        "best_epoch": best["epoch"], "best_checkpoint": summary["best_checkpoint"]})
        rows.append(row)
    groups = []
    identities = sorted({(row["candidate_name"], row["cohort"], row["phase"]) for row in rows})
    for candidate, cohort, phase in identities:
        members = [row for row in rows if (row["candidate_name"], row["cohort"], row["phase"]) == (candidate, cohort, phase)]
        complete = [row for row in members if row["status"] == "complete" and row.get("miou") is not None]
        seeds = [row["seed"] for row in complete]
        if len(seeds) != len(set(seeds)):
            raise ValueError("A candidate/cohort/phase cannot repeat a seed")
        group = {"candidate_name": candidate, "cohort": cohort, "phase": phase, "seeds": seeds,
                 "completed_runs": len(complete), "planned_runs": len(members),
                 "confirmation_complete": phase == "confirmation" and len(complete) == len(members) and len(seeds) >= 3}
        if complete:
            values = [row["miou"] for row in complete]
            group.update({"miou_mean": statistics.mean(values), "miou_population_std": statistics.pstdev(values),
                          "miou_min": min(values), "miou_max": max(values),
                          "mf1_mean": statistics.mean(row["mf1"] for row in complete),
                          "class_iou_mean": np.mean([row["class_iou"] for row in complete], axis=0).tolist(),
                          "training_seconds_total": sum(row["training_seconds"] for row in complete),
                          "peak_vram_gib": max(row["peak_vram_gib"] for row in complete)})
        groups.append(group)
    groups.sort(key=lambda row: (row["cohort"], row["phase"], -row.get("miou_mean", -1)))
    return {"schema_version": 1, "split": "val", "class_names": CLASS_NAMES, "data": plan["data"],
            "phase": plan["phase"], "rare_class_ids": rare, "runs": rows, "candidate_groups": groups,
            "unavailable_candidates": plan.get("unavailable_candidates", []),
            "selection_note": "Native public-validation summaries; keep cohorts separate and verify final checkpoints with serial prediction/evaluation before choosing submission"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    calibration = commands.add_parser("calibrate", help="Bounded real-loader native-loss timing; discard probe model")
    calibration.add_argument("--config", required=True)
    calibration.add_argument("--manifest", required=True)
    calibration.add_argument("--report", required=True)
    calibration.add_argument("--device", default="cuda:0")
    calibration.add_argument("--warmup", type=int, default=1)
    calibration.add_argument("--steps", type=int, default=3)
    calibration.add_argument("--val-images", type=int, default=3)
    calibration.add_argument("--allow-pretrained-downloads", action="store_true", help="Enable pretrained initialization from the existing cache; network remains offline")
    planning = commands.add_parser("plan", help="Write fixed per-candidate budgets and configurations")
    planning.add_argument("--calibrations", nargs="+", required=True)
    planning.add_argument("--output-dir", required=True)
    planning.add_argument("--seconds", type=float, help="Equal train plus validation timing estimate per run; default bounded 5..30 minutes")
    planning.add_argument("--evaluations", type=int, default=10)
    planning.add_argument("--phase", choices=("pilot", "confirmation"), default="pilot")
    planning.add_argument("--seeds", nargs="+", type=int)
    execution = commands.add_parser("run", help="Execute a fixed sequential queue with durable per-run logs")
    execution.add_argument("--plan", required=True)
    execution.add_argument("--device", default="cuda:0")
    execution.add_argument("--names", nargs="+")
    execution.add_argument("--fail-fast", action="store_true")
    collection = commands.add_parser("aggregate", help="Collect all-class native public validation summaries")
    collection.add_argument("--plan", required=True)
    collection.add_argument("--audit")
    collection.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    if args.command == "calibrate":
        result = calibrate(args.config, args.manifest, args.report, device_name=args.device,
                           warmup=args.warmup, steps=args.steps, val_images=args.val_images,
                           allow_pretrained_downloads=args.allow_pretrained_downloads)
        print(json.dumps(result, indent=2), flush=True)
        if result["status"] != "ok":
            raise SystemExit(1)
    elif args.command == "plan":
        print(write_plan(args.calibrations, args.output_dir, seconds=args.seconds,
                         evaluations=args.evaluations, phase=args.phase, seeds=args.seeds))
    elif args.command == "run":
        result = run_plan(args.plan, device_name=args.device, names=args.names, fail_fast=args.fail_fast)
        if result["status"] != "complete":
            raise SystemExit(1)
    else:
        result = aggregate(args.plan, audit_path=args.audit)
        output = safe_output_path(args.report, require_workspace())
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(output, result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
