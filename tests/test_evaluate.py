import numpy as np
import pytest
from PIL import Image

from palmcity.evaluate import evaluate_predictions


def test_saved_validation_predictions_score_void_and_absent_classes(tmp_path):
    target = np.array([[0, 0, 31, 31]], dtype=np.uint8)
    truth = tmp_path / "truth.png"
    Image.fromarray(target).save(truth)
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    Image.fromarray(target).save(predictions / "sample.png")
    manifest = {"train": [], "val": [{"id": "sample", "mask": str(truth)}], "test": []}
    result = evaluate_predictions(manifest, predictions)
    assert result["miou_percent"] == 6.25  # Two perfect classes divided by all 32.
    assert result["mf1_percent"] == 6.25
    assert result["classes"][31]["target_pixels"] == 2
    with pytest.raises(ValueError, match="public train or val"):
        evaluate_predictions(manifest, predictions, "test")
    Image.fromarray(target).save(predictions / "extra.png")
    with pytest.raises(ValueError, match="exactly one PNG"):
        evaluate_predictions(manifest, predictions)
