"""Run a fixed, resumable validation comparison without dataset probability files."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import torch

from .benchmark import idle_gpu
from .compare import comparison
from .data import CLASS_NAMES, data_input_sha256, prediction_set_sha256, read_manifest
from .evaluate import evaluate_predictions
from .inference import InferenceOptions
from .predict import predict
from .storage import require_free_space, require_workspace, safe_output_path
from .training_state import atomic_json, code_identity, dependency_versions, file_sha256, json_sha256


def default_plan(winner_checkpoint: str | Path, runnerup_checkpoint: str | Path,
                 winner_seed_checkpoints: list[str | Path], *, include_window: bool = True) -> dict:
    """Declare three ensemble weights and TTA branches before inspecting scores."""
    if len(winner_seed_checkpoints) != 3:
        raise ValueError("The default comparison requires exactly three winner seed checkpoints")
    winner, runner = str(winner_checkpoint), str(runnerup_checkpoint)
    ensemble_names = ["winner-seeds-equal", "cross-equal", "cross-winner75", "cross-runner75"]
    variants = [
        {"name": "winner-single", "checkpoints": [winner]},
        {"name": "runner-single", "checkpoints": [runner]},
        {"name": ensemble_names[0], "checkpoints": list(map(str, winner_seed_checkpoints))},
        {"name": ensemble_names[1], "checkpoints": [winner, runner], "weights": [0.5, 0.5]},
        {"name": ensemble_names[2], "checkpoints": [winner, runner], "weights": [0.75, 0.25]},
        {"name": ensemble_names[3], "checkpoints": [winner, runner], "weights": [0.25, 0.75]},
        {"name": "winner-flip", "checkpoints": [winner], "hflip_tta": True},
        {"name": "winner-multiscale-flip", "checkpoints": [winner], "hflip_tta": True, "scales": [0.75, 1.0, 1.25]},
        {"name": "selected-ensemble-flip", "select_best_of": ensemble_names, "hflip_tta": True},
        {"name": "selected-ensemble-multiscale-flip", "select_best_of": ensemble_names,
         "hflip_tta": True, "scales": [0.75, 1.0, 1.25]},
    ]
    if include_window:
        variants.append({"name": "winner-window", "checkpoints": [winner],
                         "window_size": [256, 512], "overlap": 0.5})
    return {"variants": variants,
            "selection_scope": "Public validation only; four declared ensembles select the two TTA branches"}


def _validated_plan(payload: Any, base: Path) -> list[dict]:
    variants = payload.get("variants") if isinstance(payload, dict) else payload
    if not isinstance(variants, list) or not 1 <= len(variants) <= 12:
        raise ValueError("Declare a fixed list of 1..12 validation variants")
    result, names = [], set()
    allowed = {"name", "checkpoints", "select_best_of", "weights", "run_dirs", "hflip_tta", "scales",
               "window_size", "overlap", "amp", "max_scaled_pixels", "max_tiles", "estimated_seconds"}
    for original in variants:
        if not isinstance(original, dict) or set(original) - allowed:
            raise ValueError("Variant must be an object with recognized inference fields")
        variant = copy.deepcopy(original)
        name = variant.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in names:
            raise ValueError("Variant names must be unique plain directory names")
        sources = variant.get("select_best_of")
        checkpoints = variant.get("checkpoints")
        if (sources is None) == (checkpoints is None):
            raise ValueError("Each variant needs checkpoints or select_best_of, exclusively")
        if sources is not None:
            if not isinstance(sources, list) or not sources or any(source not in names for source in sources):
                raise ValueError("select_best_of must refer only to earlier declared variants")
            if "weights" in variant or "run_dirs" in variant:
                raise ValueError("Selected variants inherit ensemble weights and run directories")
        else:
            if not isinstance(checkpoints, list) or not checkpoints or len(checkpoints) > 4:
                raise ValueError("A validation variant needs 1..4 local checkpoints")
            variant["checkpoints"] = [str((base / Path(path).expanduser()).resolve()) for path in checkpoints]
            weights = variant.get("weights", [1.0] * len(checkpoints))
            if len(weights) != len(checkpoints) or not all(math.isfinite(w) and w >= 0 for w in weights) or not math.isfinite(sum(weights)) or sum(weights) <= 0:
                raise ValueError("Variant weights must be finite, nonnegative and match checkpoints")
            variant["weights"] = [w / sum(weights) for w in weights]
        if "run_dirs" in variant:
            variant["run_dirs"] = [str((base / Path(path).expanduser()).resolve()) for path in variant["run_dirs"]]
        InferenceOptions(tuple(variant.get("scales", [1.0])),
                         tuple(variant["window_size"]) if variant.get("window_size") else None,
                         variant.get("overlap", 0.5), variant.get("max_scaled_pixels", 2_097_152),
                         variant.get("max_tiles", 512))
        if "estimated_seconds" in variant and (not math.isfinite(variant["estimated_seconds"]) or variant["estimated_seconds"] <= 0):
            raise ValueError("estimated_seconds must be finite and positive")
        for key in ("hflip_tta", "amp"):
            if key in variant and not isinstance(variant[key], bool):
                raise ValueError(f"{key} must be boolean")
        names.add(name)
        result.append(variant)
    return result


def _resolved_variant(variant: dict, state: dict) -> dict:
    resolved = copy.deepcopy(variant)
    if "select_best_of" not in resolved:
        return resolved
    candidates = [state["variants"][name] for name in resolved.pop("select_best_of")
                  if state["variants"][name]["status"] == "complete"]
    if not candidates:
        raise ValueError("No completed source ensemble is available for the declared TTA branch")
    source = min(candidates, key=lambda row: (-row["miou"], row["prediction_seconds"]))
    inherited = source["attempts"][-1]["resolved_variant"]
    for key in ("checkpoints", "weights", "run_dirs"):
        if key in inherited:
            resolved[key] = inherited[key]
    resolved["selected_source"] = source["name"]
    return resolved


def _experiment(row: dict) -> dict:
    attempt = row["attempts"][-1]
    resolved = attempt["resolved_variant"]
    result = {"name": row["name"], "score_report": attempt["score_report"],
              "prediction_report": attempt["prediction_report"]}
    run_dirs = resolved.get("run_dirs")
    if run_dirs is None:
        run_dirs = sorted({str(Path(path).parent) for path in resolved["checkpoints"]
                           if (Path(path).parent / "metadata.json").is_file()
                           and (Path(path).parent / "summary.json").is_file()})
    if run_dirs:
        result["run_dirs"] = run_dirs
    return result


def run_sweep(plan_path: str | Path, manifest_path: str | Path, audit_path: str | Path,
              output_dir: str | Path, *, device_name: str = "cuda:1", budget_seconds: float = 1800,
              resume: bool = False, retry_failed: bool = False) -> Path:
    """Resume verified results; preserve partial attempts and admit at most 12 variants.

    The time budget accumulates across calls. Inference checks it between images;
    checkpoint reconstruction or one image can overrun the deadline. This is a
    declared validation comparison, not an adaptive hyperparameter search.
    """
    if not math.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError("budget_seconds must be finite and positive")
    workspace = require_workspace()
    output = safe_output_path(output_dir, workspace)
    state_path, report_path = output / "state.json", output / "comparison.json"
    if output.exists() and not resume:
        raise FileExistsError(f"Sweep directory already exists: {output}")
    if resume and not state_path.is_file():
        raise ValueError("Resume requires an existing state.json in this sweep directory")
    plan_path = Path(plan_path).resolve(strict=True)
    variants = _validated_plan(json.loads(plan_path.read_text()), plan_path.parent)
    manifest = read_manifest(manifest_path)
    audit = json.loads(Path(audit_path).read_text())
    manifest_digest = file_sha256(manifest_path)
    val_digest = data_input_sha256(manifest, splits=("val",))
    if not manifest["val"] or not audit.get("valid") or audit.get("class_names") != CLASS_NAMES:
        raise ValueError("A valid official-class dataset audit and nonempty public val split are required")
    if audit.get("manifest_sha256") != manifest_digest or audit.get("validation_input_sha256") != val_digest:
        raise ValueError("Dataset audit differs from validation manifest/input content")
    checkpoints = sorted({path for variant in variants for path in variant.get("checkpoints", [])})
    identity = {"plan_sha256": json_sha256(variants), "manifest_sha256": manifest_digest,
                "audit_sha256": file_sha256(audit_path), "validation_input_sha256": val_digest,
                "checkpoints": {path: file_sha256(path) if Path(path).is_file() else None for path in checkpoints},
                "code": code_identity(), "dependencies": dependency_versions(), "device": device_name}
    if resume:
        state = json.loads(state_path.read_text())
        if state.get("identity") != identity:
            raise ValueError("Sweep resume identity differs: plan, checkpoint, data, code, dependencies or device changed")
        for row in state["variants"].values():
            if row["status"] == "running":
                interrupted = row["attempts"][-1]
                if "elapsed_seconds" not in interrupted:
                    # A hard process termination cannot commit its elapsed time.
                    # Conservatively charge its reserved remainder, preventing
                    # repeated interruptions from silently expanding the budget.
                    state["elapsed_seconds"] += interrupted.get("reserved_seconds", 0)
                interrupted["status"] = "interrupted"
                row["status"] = "budget_deferred"
            if row["status"] == "complete":
                attempt = row["attempts"][-1]
                if any(file_sha256(attempt[key]) != attempt[key + "_sha256"]
                       for key in ("score_report", "prediction_report")):
                    raise ValueError("A completed sweep report changed since commit")
                if prediction_set_sha256(attempt["prediction_dir"], manifest["val"]) != attempt["prediction_set_sha256"]:
                    raise ValueError("Completed validation PNGs changed since commit")
    else:
        state = {"schema_version": 1, "identity": identity, "elapsed_seconds": 0.0,
                 "variants": {v["name"]: {"name": v["name"], "status": "pending", "attempts": []} for v in variants}}
    require_free_space(workspace, gib=0.1)
    output.mkdir(parents=True, exist_ok=True)
    state["budget_seconds"] = budget_seconds
    atomic_json(state_path, state)
    device_spec = torch.device(device_name)
    pending = [state["variants"][v["name"]] for v in variants
               if state["variants"][v["name"]]["status"] != "complete"
               and (state["variants"][v["name"]]["status"] != "failed" or retry_failed)]
    if pending and state["elapsed_seconds"] < budget_seconds and device_spec.type == "cuda":
        if device_spec.index is None:
            raise ValueError("GPU validation sweeps require explicit physical cuda:N")
        state["gpu_preflight"] = idle_gpu(device_spec.index)
        atomic_json(state_path, state)
    call_started, elapsed_before = time.monotonic(), state["elapsed_seconds"]
    for variant in variants:
        row = state["variants"][variant["name"]]
        if row["status"] == "complete" or (row["status"] == "failed" and not retry_failed):
            continue
        state["elapsed_seconds"] = elapsed_before + time.monotonic() - call_started
        remaining = budget_seconds - state["elapsed_seconds"]
        if remaining <= 0 or variant.get("estimated_seconds", 0) > remaining:
            row["status"] = "budget_deferred"
            atomic_json(state_path, state)
            continue
        attempt_number = len(row["attempts"]) + 1
        attempt_dir = output / variant["name"] / f"attempt-{attempt_number:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        attempt = {"attempt": attempt_number, "status": "running", "prediction_dir": str(attempt_dir / "png"),
                   "prediction_report": str(attempt_dir / "png.json"), "score_report": str(attempt_dir / "score.json"),
                   "reserved_seconds": remaining}
        row["attempts"].append(attempt)
        row["status"] = "running"
        atomic_json(state_path, state)
        tick = time.monotonic()
        try:
            resolved = _resolved_variant(variant, state)
            attempt["resolved_variant"] = resolved
            kwargs = {key: resolved[key] for key in ("weights", "hflip_tta", "scales", "overlap", "amp", "max_scaled_pixels", "max_tiles") if key in resolved}
            if resolved.get("window_size"):
                kwargs["window_size"] = tuple(resolved["window_size"])
            predict(resolved["checkpoints"], manifest_path, attempt["prediction_dir"], split="val",
                    device_name=device_name, time_limit_seconds=max(0.001, remaining), **kwargs)
            score = evaluate_predictions(manifest, attempt["prediction_dir"], "val")
            score["manifest_sha256"] = manifest_digest
            atomic_json(Path(attempt["score_report"]), score)
            inference = json.loads(Path(attempt["prediction_report"]).read_text())
            row.update({"status": "complete", "miou": score["miou"], "mf1": score["mf1"],
                        "prediction_seconds": inference["runtime"]["prediction_seconds"]})
            attempt.update({"status": "complete", "prediction_set_sha256": score["prediction_set_sha256"],
                            "score_report_sha256": file_sha256(attempt["score_report"]),
                            "prediction_report_sha256": file_sha256(attempt["prediction_report"])})
            report = comparison([_experiment(r) for r in state["variants"].values() if r["status"] == "complete"], audit)
            atomic_json(report_path, report)
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
            attempt.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
            row["status"] = "budget_deferred" if isinstance(error, TimeoutError) else "failed"
        finally:
            attempt["elapsed_seconds"] = time.monotonic() - tick
            state["elapsed_seconds"] = elapsed_before + time.monotonic() - call_started
            atomic_json(state_path, state)
        print(json.dumps({"variant": row["name"], "status": row["status"], "seconds": attempt["elapsed_seconds"],
                          "miou": row.get("miou"), "error": attempt.get("error")}), flush=True)
    completed = [row for row in state["variants"].values() if row["status"] == "complete"]
    if completed:
        # comparison.json is derived; reconstruct it on a resumed completed run
        # even if a previous process failed before its aggregate report commit.
        atomic_json(report_path, comparison([_experiment(row) for row in completed], audit))
    state["status"] = "complete" if len(completed) == len(variants) else "partial"
    state["completed_variants"] = len(completed)
    state["quality_metrics_measured"] = bool(completed)
    state["selected_variant"] = min(completed, key=lambda row: (-row["miou"], row["prediction_seconds"]))["name"] if completed else None
    state["comparison_report"] = str(report_path) if report_path.exists() else None
    state["elapsed_seconds"] = elapsed_before + time.monotonic() - call_started
    atomic_json(state_path, state)
    return state_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--budget-seconds", type=float, default=1800)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    state_path = run_sweep(args.plan, args.manifest, args.audit, args.output_dir, device_name=args.device,
                           budget_seconds=args.budget_seconds, resume=args.resume, retry_failed=args.retry_failed)
    print(state_path)


if __name__ == "__main__":
    main()
