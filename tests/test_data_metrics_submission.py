"""Integrity checks for decoding, official scoring, and competition archives."""

import io
import json
import zipfile

import numpy as np
from PIL import Image
import pytest

from palmcity.data import PALETTE, audit_manifest, build_manifest, load_mask, read_manifest
from palmcity.metrics import confusion_matrix, scores_from_confusion
from palmcity import submission


def write_png(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(values, dtype=np.uint8)).save(path)
    return path


def manifest_for_test(tmp_path):
    image = write_png(tmp_path / "images" / "sample.png", np.zeros((2, 3, 3), dtype=np.uint8))
    return {"train": [], "val": [], "test": [{"id": "sample", "image": str(image)}]}


def png_bytes(values):
    stream = io.BytesIO()
    Image.fromarray(np.asarray(values, dtype=np.uint8)).save(stream, format="PNG")
    return stream.getvalue()


def test_grayscale_and_rgb_decode_same_all_32_ids(tmp_path):
    ids = np.arange(32, dtype=np.uint8).reshape(4, 8)
    grayscale = write_png(tmp_path / "gray.png", ids)
    rgb = write_png(tmp_path / "rgb.png", PALETTE[ids])
    np.testing.assert_array_equal(load_mask(grayscale), ids)
    np.testing.assert_array_equal(load_mask(rgb), ids)
    assert load_mask(rgb).dtype == np.uint8


@pytest.mark.parametrize("kind", ["invalid_id", "unknown_color", "transparent"])
def test_invalid_dataset_masks_rejected(tmp_path, kind):
    path = tmp_path / "bad.png"
    if kind == "invalid_id":
        values = np.array([[32]], dtype=np.uint8)
    elif kind == "unknown_color":
        values = np.array([[[128, 64, 127]]], dtype=np.uint8)
    else:
        values = np.array([[[128, 64, 128, 255]]], dtype=np.uint8)
    write_png(path, values)
    with pytest.raises(ValueError):
        load_mask(path)


def test_indexed_palette_requires_unambiguous_class_indices(tmp_path):
    ids = np.array([[0, 31]], dtype=np.uint8)
    canonical = Image.fromarray(ids).convert("P")
    canonical.putpalette(np.pad(PALETTE, ((0, 224), (0, 0))).ravel().tolist())
    path = tmp_path / "indexed.png"
    canonical.save(path)
    np.testing.assert_array_equal(load_mask(path), ids)
    gray = Image.fromarray(ids).convert("P")
    gray_path = tmp_path / "indexed-gray.png"
    gray.save(gray_path)
    np.testing.assert_array_equal(load_mask(gray_path), ids)
    wrong_palette = np.pad(PALETTE, ((0, 224), (0, 0)))
    wrong_palette[[0, 31]] = wrong_palette[[31, 0]]
    canonical.putpalette(wrong_palette.ravel().tolist())
    canonical.save(tmp_path / "ambiguous.png")
    with pytest.raises(ValueError, match="ambiguous indexed palette"):
        load_mask(tmp_path / "ambiguous.png")


def test_build_manifest_uses_explicit_mask_template(tmp_path):
    for index, split in enumerate(("train", "val", "test")):
        write_png(tmp_path / "images" / split / f"image{index}.png", np.full((2, 3, 3), index, dtype=np.uint8))
        if split != "test":
            write_png(tmp_path / "annotations" / "gt" / split / f"image{index}_labels.png", np.full((2, 3), index, dtype=np.uint8))
    with pytest.raises(ValueError, match="Missing mask"):
        build_manifest(tmp_path)
    manifest = build_manifest(tmp_path, mask_template="{stem}_labels.png")
    assert manifest["train"][0]["mask"].endswith("image0_labels.png")
    assert "mask" not in manifest["test"][0]
    report = audit_manifest(manifest, expected_size=(3, 2))
    assert report["valid"]
    assert report["class_histograms"]["train"][0] == 6
    assert report["class_histograms"]["val"][1] == 6
    counts_report = audit_manifest(manifest, expected_size=(3, 2), strict_counts=True)
    assert not counts_report["valid"]
    assert any("expected 497" in error for error in counts_report["errors"])


def test_audit_detects_decoded_image_leakage_and_shape_errors(tmp_path):
    train_image = write_png(tmp_path / "train.png", np.full((2, 3, 3), 10, dtype=np.uint8))
    val_image = write_png(tmp_path / "val.png", np.full((2, 3, 3), 10, dtype=np.uint8))
    mask = write_png(tmp_path / "mask.png", np.zeros((1, 3), dtype=np.uint8))
    manifest = {
        "train": [{"id": "train", "image": str(train_image), "mask": str(mask)}],
        "val": [{"id": "val", "image": str(val_image), "mask": str(mask)}],
        "test": [],
    }
    report = audit_manifest(manifest, expected_size=(3, 2))
    assert not report["valid"]
    assert len(report["duplicate_images"]) == 1
    assert any("across splits (leakage)" in error for error in report["errors"])
    assert any("mask shape" in error for error in report["errors"])


@pytest.mark.parametrize("change", ["relative", "duplicate", "mismatch", "path_id"])
def test_manifest_rejects_ambiguous_records(tmp_path, change):
    manifest = manifest_for_test(tmp_path)
    if change == "relative":
        manifest["test"][0]["image"] = "sample.png"
    elif change == "duplicate":
        manifest["test"].append(manifest["test"][0].copy())
    elif change == "mismatch":
        manifest["test"][0]["id"] = "wrong"
    else:
        manifest["test"][0]["id"] = "../sample"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        read_manifest(path)


def test_official_metrics_include_void_and_zero_absent_classes():
    target = np.array([[0, 31], [31, 2]], dtype=np.uint8)
    prediction = np.array([[0, 31], [0, 3]], dtype=np.uint8)
    counts = confusion_matrix(target, prediction)
    assert counts.dtype == np.int64
    assert counts[31, 0] == 1
    scores = scores_from_confusion(counts)
    assert scores["iou"][0] == pytest.approx(0.5)
    assert scores["iou"][31] == pytest.approx(0.5)
    assert scores["iou"][30] == 0
    assert scores["miou"] == pytest.approx(1 / 32)
    assert scores["mf1"] == pytest.approx(1 / 24)
    perfect_void = scores_from_confusion(confusion_matrix(np.array([31]), np.array([31])))
    assert perfect_void["miou"] == pytest.approx(1 / 32)
    assert perfect_void["mf1"] == pytest.approx(1 / 32)


def test_metrics_aggregate_pixel_counts_before_scoring():
    first = confusion_matrix(np.array([0]), np.array([0]))
    second = confusion_matrix(np.zeros(9, dtype=np.uint8), np.ones(9, dtype=np.uint8))
    scores = scores_from_confusion(first + second)
    assert scores["miou"] == pytest.approx(0.1 / 32)
    assert scores["miou"] != pytest.approx((scores_from_confusion(first)["miou"] + scores_from_confusion(second)["miou"]) / 2)


@pytest.mark.parametrize("target,prediction", [([32], [0]), ([0], [-1]), ([0.5], [0]), ([0, 1], [0])])
def test_metrics_reject_invalid_inputs(target, prediction):
    with pytest.raises(ValueError):
        confusion_matrix(np.asarray(target), np.asarray(prediction))


def test_submission_is_flat_valid_and_refuses_overwrite(tmp_path, monkeypatch):
    manifest = manifest_for_test(tmp_path)
    predictions = tmp_path / "predictions"
    write_png(predictions / "sample.png", np.full((2, 3), 31, dtype=np.uint8))
    monkeypatch.setattr(submission, "require_workspace", lambda: tmp_path)
    monkeypatch.setattr("palmcity.storage.require_workspace", lambda: tmp_path)
    output = submission.create_submission(manifest, predictions, "submission.zip", expected_size=(3, 2), allow_partial=True)
    assert output == tmp_path / "submission.zip"
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == ["sample.png"]
    submission.validate_submission_zip(output, manifest, expected_size=(3, 2), allow_partial=True)
    with pytest.raises(ValueError, match="overwrite"):
        submission.create_submission(manifest, predictions, "submission.zip", expected_size=(3, 2), allow_partial=True)


@pytest.mark.parametrize("kind", ["missing", "extra", "rgb", "size", "invalid_id", "nested"])
def test_prediction_errors_prevent_submission_creation(tmp_path, monkeypatch, kind):
    manifest = manifest_for_test(tmp_path)
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    if kind != "missing":
        values = np.zeros((2, 3), dtype=np.uint8)
        if kind == "rgb":
            values = PALETTE[values]
        elif kind == "size":
            values = np.zeros((3, 2), dtype=np.uint8)
        elif kind == "invalid_id":
            values[:] = 32
        write_png(predictions / "sample.png", values)
    if kind == "extra":
        write_png(predictions / "extra.png", np.zeros((2, 3), dtype=np.uint8))
    if kind == "nested":
        write_png(predictions / "folder" / "sample.png", np.zeros((2, 3), dtype=np.uint8))
    monkeypatch.setattr(submission, "require_workspace", lambda: tmp_path)
    monkeypatch.setattr("palmcity.storage.require_workspace", lambda: tmp_path)
    with pytest.raises(ValueError):
        submission.create_submission(manifest, predictions, "bad.zip", expected_size=(3, 2), allow_partial=True)
    assert not (tmp_path / "bad.zip").exists()


@pytest.mark.parametrize("kind", ["missing", "extra", "duplicate", "nested", "traversal", "rgb", "invalid_id"])
def test_zip_validation_rejects_invalid_archives(tmp_path, kind):
    manifest = manifest_for_test(tmp_path)
    values = np.full((2, 3), 31, dtype=np.uint8)
    if kind == "rgb":
        values = PALETTE[values]
    elif kind == "invalid_id":
        values[:] = 32
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "x") as archive:
        if kind != "missing":
            name = {"nested": "folder/sample.png", "traversal": "../sample.png"}.get(kind, "sample.png")
            archive.writestr(name, png_bytes(values))
        if kind == "extra":
            archive.writestr("extra.png", png_bytes(values))
        if kind == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr("sample.png", png_bytes(values))
    with pytest.raises(ValueError):
        submission.validate_submission_zip(archive_path, manifest, expected_size=(3, 2), allow_partial=True)


def test_official_submission_rejects_partial_test_manifest(tmp_path):
    manifest = manifest_for_test(tmp_path)
    predictions = tmp_path / "predictions"
    write_png(predictions / "sample.png", np.zeros((2, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="requires 249"):
        submission.validate_predictions(predictions, manifest, expected_size=(3, 2))
