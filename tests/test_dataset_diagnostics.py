from pathlib import Path

import numpy as np
from PIL import Image

from palmcity.dataset_diagnostics import diagnose


def test_support_and_neighbor_checks_preserve_split_without_test_labels(tmp_path: Path):
    manifest = {}
    for index, split in enumerate(("train", "val", "test")):
        identifier = f"GS__{index:04}"
        image = tmp_path / (identifier + ".png")
        Image.new("RGB", (4, 2), (20 * index, 20 * index, 20 * index)).save(image)
        record = {"id": identifier, "image": str(image)}
        if split != "test":
            mask = tmp_path / (identifier + "_gt.png")
            Image.fromarray(np.full((2, 4), 31 if split == "train" else 0, dtype=np.uint8)).save(mask)
            record["mask"] = str(mask)
        manifest[split] = [record]
    result = diagnose(manifest)
    assert "test" not in result["label_support"]
    assert result["label_support"]["train"]["classes"][31]["image_count"] == 1
    assert result["label_support"]["val"]["classes"][31]["image_count"] == 0
    assert len(result["neighboring_filename_candidates"]) == 2
    assert result["nearest_train_thumbnail_for_each_val"] == [
        {"val_id": "GS__0001", "train_id": "GS__0000", "thumbnail_rgb_rmse_0_255": 20.0}]
