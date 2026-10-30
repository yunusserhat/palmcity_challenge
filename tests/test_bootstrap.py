from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from palmcity.bootstrap import image_confusions, paired_bootstrap
from palmcity.metrics import scores_from_confusion


def _matrices(seed: int = 0, images: int = 6) -> np.ndarray:
    return np.random.default_rng(seed).integers(1, 50, size=(images, 32, 32), dtype=np.int64)


def test_aggregate_scores_match_official_metric():
    matrices = _matrices()
    result = paired_bootstrap(matrices, samples=100)
    official = scores_from_confusion(matrices.sum(axis=0))
    assert result["miou"] == pytest.approx(official["miou"], abs=1e-15)
    assert result["mf1"] == pytest.approx(official["mf1"], abs=1e-15)
    assert [row["iou"] for row in result["classes"]] == pytest.approx(official["iou"], abs=1e-15)


def test_identical_predictions_have_zero_paired_difference():
    matrices = _matrices(1)
    candidate = paired_bootstrap(matrices, matrices.copy(), samples=200, seed=3)["candidate"]
    assert candidate["delta_miou"] == 0
    assert candidate["bootstrap_positive_fraction"] == 0
    assert candidate["paired_bootstrap_delta_miou"]["percentile_2_5"] == 0
    assert candidate["paired_bootstrap_delta_miou"]["percentile_97_5"] == 0


def test_resampling_is_paired_and_seeded():
    baseline = _matrices(2)
    better = baseline.copy()
    better[:, 0, 0] += better[:, 0, 1]
    better[:, 0, 1] = 0
    first = paired_bootstrap(baseline, better, samples=300, seed=7)
    assert first == paired_bootstrap(baseline, better, samples=300, seed=7)
    assert first["candidate"]["bootstrap_positive_fraction"] == 1.0
    assert first["candidate"]["delta_miou"] > 0
    assert first != paired_bootstrap(baseline, better, samples=300, seed=8)


@pytest.mark.parametrize("samples", [99, 20001])
def test_invalid_sample_counts_are_rejected(samples):
    with pytest.raises(ValueError, match="bootstrap samples"):
        paired_bootstrap(_matrices(), samples=samples)


def test_image_confusions_require_the_exact_png_set(tmp_path: Path):
    records = []
    for name in ("a", "b"):
        mask = np.zeros((4, 8), dtype=np.uint8)
        mask[:, 4:] = 31
        Image.fromarray(mask).save(tmp_path / f"{name}-mask.png")
        records.append({"id": name, "mask": str(tmp_path / f"{name}-mask.png")})
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    Image.fromarray(np.zeros((4, 8), dtype=np.uint8)).save(predictions / "a.png")
    with pytest.raises(ValueError, match="exactly one PNG"):
        image_confusions(records, predictions)
    Image.fromarray(np.full((4, 8), 31, dtype=np.uint8)).save(predictions / "b.png")
    matrices = image_confusions(records, predictions)
    assert matrices.shape == (2, 32, 32)
    assert matrices[0, 0, 0] == 16 and matrices[0, 31, 0] == 16
    assert matrices[1, 0, 31] == 16 and matrices[1, 31, 31] == 16
