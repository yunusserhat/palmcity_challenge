import pytest

from palmcity.benchmark import proposed_budget


def test_budget_includes_validation_and_accumulation():
    report = {"status": "ok", "batch_size": 2, "gradient_accumulation": 2,
              "drop_last": True, "training_microbatch_seconds_median": 0.5,
              "inference_seconds_per_image": 0.1}
    result = proposed_budget(report, hours=1, train_images=497, val_images=84)
    assert result["effective_batch_size"] == 4
    assert result["estimated_updates_per_epoch"] == 124
    assert result["estimated_validation_seconds_per_epoch"] == pytest.approx(8.4)
    assert result["max_optimizer_steps"] == int(3600 / (1 + 8.4 / 124))
    assert result["estimated_total_seconds"] <= 3600
    assert result["epochs"] * 124 >= result["max_optimizer_steps"]


@pytest.mark.parametrize("status", ["oom", "error", "initializing"])
def test_failed_probe_cannot_propose_budget(status):
    with pytest.raises(ValueError, match="successful probe"):
        proposed_budget({"status": status})
