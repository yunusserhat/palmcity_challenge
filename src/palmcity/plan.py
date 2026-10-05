"""Write conditional experiment budgets from probes; never launch training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .benchmark import proposed_budget
from .storage import require_workspace, safe_output_path


def plan_budgets(reports: list[dict], *, evaluations: int = 10) -> dict:
    if not reports or not 2 <= evaluations <= 10:
        raise ValueError("Need successful probes and 2..10 reserved validation opportunities")
    if any(r.get("schema_version") != 2 or r.get("status") != "ok" for r in reports):
        raise ValueError("Use successful version-2 accumulation and serial-inference probes")
    conditions = {(tuple(r["image_size"]), r["batch_size"] * r["gradient_accumulation"], r["amp_dtype"])
                  for r in reports}
    if len(conditions) != 1:
        raise ValueError("All pilots need the same resolution, effective batch and AMP dtype")
    probe_budgets = [proposed_budget(r) for r in reports]
    slowest_epoch = max(b["estimated_updates_per_epoch"] * r["training_update_seconds_median"]
                        + 84 * r["inference_seconds_per_image"] for r, b in zip(reports, probe_budgets, strict=True))
    pilot_seconds = max(300.0, min(1800.0, evaluations * slowest_epoch))
    rows = []
    for report, budget in zip(reports, probe_budgets, strict=True):
        validation_seconds = 84 * report["inference_seconds_per_image"] * evaluations
        updates = max(1, int((pilot_seconds - validation_seconds) / report["training_update_seconds_median"]))
        epochs = math.ceil(updates / budget["estimated_updates_per_epoch"])
        if epochs < evaluations:
            raise ValueError("30-minute pilot cap cannot support the declared validation opportunities; revise candidates/budget")
        validation_epochs = [math.ceil(i * epochs / evaluations) for i in range(1, evaluations + 1)]
        rows.append({"config_path": report["config_path"], "max_optimizer_steps": updates,
                     "epochs": epochs, "validation_epochs": validation_epochs,
                     "warmup_steps": max(1, int(0.05 * updates)),
                     "target_model_compute_seconds": pilot_seconds,
                     "estimated_total_model_compute_seconds": updates * report["training_update_seconds_median"] + validation_seconds})
    return {"status": "conditional_not_executed", "basis": "random synthetic probes; recalibrate real IO and pretrained recipe before training",
            "blockers": ["writable permanent storage", "independent backup target", "individual checkpoint license/access validation"],
            "pilot_seed": 42, "pilot_configs": rows, "pilot_seconds_per_candidate": pilot_seconds,
            "reserved_validation_evaluations": evaluations,
            "confirmation": {"top_candidates": 2, "seeds": [42, 123, 2026],
                             "seconds_per_run": 4 * pilot_seconds, "validation_evaluations": evaluations},
            "inference_comparison_gpu_hours_cap": 0.5,
            "total_model_compute_gpu_hours_cap_per_initialization_cohort": (len(reports) + 2 * 3 * 4) * pilot_seconds / 3600 + 0.5,
            "initialization_policy": "Separate random and pretrained cohorts; all generated configs remain random until weight/source review",
            "third_candidate": "Only if complementary class errors justify a separately budgeted expansion"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    directory = safe_output_path(args.output_dir, require_workspace())
    if directory.exists():
        raise FileExistsError(directory)
    reports = [json.loads(Path(path).read_text()) for path in args.probes]
    result = plan_budgets(reports)
    names = [Path(row["config_path"]).stem for row in result["pilot_configs"]]
    if len(names) != len(set(names)):
        raise ValueError("Config names must be unique")
    directory.mkdir(parents=True, exist_ok=False)
    for row, name in zip(result["pilot_configs"], names, strict=True):
        config = json.loads(Path(row["config_path"]).read_text())
        config.update({key: row[key] for key in ("epochs", "max_optimizer_steps", "validation_epochs", "warmup_steps")})
        config["seed"] = result["pilot_seed"]
        config.pop("validation_every_epochs", None)
        (directory / f"{name}.json").write_text(json.dumps(config, indent=2) + "\n")
    (directory / "plan.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
