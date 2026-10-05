import pytest

from palmcity.plan import plan_budgets


def report(seconds):
    return {"schema_version": 2, "status": "ok", "image_size": [512, 1024],
            "batch_size": 1, "gradient_accumulation": 4, "amp_dtype": "torch.bfloat16",
            "drop_last": False, "training_update_seconds_median": seconds,
            "training_microbatch_seconds_median": seconds / 4,
            "inference_seconds_per_image": 0.03, "config_path": "candidate.json"}


def test_equal_model_compute_with_reserved_validation_and_bounded_confirmation():
    result = plan_budgets([report(0.05), report(0.4)])
    assert result["status"] == "conditional_not_executed"
    assert result["confirmation"]["seeds"] == [42, 123, 2026]
    budget = result["pilot_seconds_per_candidate"]
    for row in result["pilot_configs"]:
        assert len(row["validation_epochs"]) == 10
        assert row["validation_epochs"] == sorted(set(row["validation_epochs"]))
        assert row["validation_epochs"][-1] == row["epochs"]
        assert row["max_optimizer_steps"] <= row["epochs"] * 125
        assert row["estimated_total_model_compute_seconds"] <= budget
        assert row["estimated_total_model_compute_seconds"] >= budget - 0.4


def test_mixed_experimental_conditions_cannot_generate_fair_plan():
    different = report(0.4)
    different["image_size"] = [256, 512]
    with pytest.raises(ValueError, match="same resolution"):
        plan_budgets([report(0.05), different])
