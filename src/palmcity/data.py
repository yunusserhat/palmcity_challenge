"""Explicit PalmCity manifests and audits; this module never downloads data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


NUM_CLASSES = 32
CLASS_NAMES = [
    "Road", "Sidewalk", "Parking Lot", "Soil", "Pedestrian", "Driver",
    "Car", "Truck", "Bus", "Motorcycle", "Bicycle", "Traffic Light",
    "Traffic Sign", "Pole", "Garbage Box", "Sitting Bench",
    "Infrastructure Cover", "Infrastructure Box", "Parking Barrier",
    "Building", "Wall", "Fence", "Stairs", "Railing", "Overpass",
    "Water Surface", "Sky", "Tree", "Grass", "Pruned Tree",
    "Operator and Shadow", "Void",
]
# Official tools/palmcity_colorize.py, read 2026-10-04:
# https://github.com/PalmCityDataset/palmcity/blob/main/tools/palmcity_colorize.py
PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [250, 170, 160], [192, 182, 154],
    [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
    [0, 60, 100], [0, 0, 230], [119, 11, 32], [250, 170, 30],
    [220, 220, 0], [153, 153, 153], [137, 145, 169], [145, 161, 153],
    [74, 68, 42], [54, 95, 145], [255, 129, 0], [70, 70, 70],
    [102, 102, 156], [190, 153, 153], [217, 149, 148], [180, 165, 180],
    [178, 161, 199], [51, 102, 255], [70, 130, 180], [107, 142, 35],
    [194, 214, 155], [0, 176, 80], [0, 0, 30], [0, 0, 0],
], dtype=np.uint8)
SPLITS = ("train", "val", "test")
OFFICIAL_COUNTS = {"train": 497, "val": 84, "test": 249}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _validate_ids(mask: np.ndarray, label: str) -> np.ndarray:
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError(f"{label}: expected a two-dimensional integer class-ID mask")
    invalid = (mask < 0) | (mask >= NUM_CLASSES)
    if invalid.any():
        raise ValueError(f"{label}: invalid class IDs {np.unique(mask[invalid]).tolist()}; expected 0..31")
    return mask.astype(np.uint8, copy=False)


def load_mask(path: str | Path) -> np.ndarray:
    """Read class IDs or decode exact official RGB colors, without guessing.

    Indexed PNGs must use class IDs as indices, with either the official
    coloring or an identity grayscale palette. A differently ordered palette
    is ambiguous and must first be explicitly converted to official RGB.
    Transparency, non-PNG files, and unknown colors are rejected.
    """
    path = Path(path)
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError(f"{path}: masks must be PNG files")
        if "transparency" in image.info or image.mode in {"RGBA", "LA", "PA"}:
            raise ValueError(f"{path}: transparent mask is ambiguous; use class IDs or opaque RGB")
        values = np.asarray(image)
        if image.mode == "P":
            mask = _validate_ids(values, str(path))
            palette = image.getpalette("RGB")
            if palette is not None:
                used = np.unique(mask)
                colors = np.asarray(palette, dtype=np.uint8).reshape(-1, 3)
                if used.max(initial=0) >= len(colors):
                    raise ValueError(f"{path}: palette has no entry for a used class ID")
                used_colors = colors[used]
                grayscale = np.repeat(used[:, None], 3, axis=1)
                if not (np.array_equal(used_colors, PALETTE[used]) or np.array_equal(used_colors, grayscale)):
                    raise ValueError(f"{path}: ambiguous indexed palette; indices do not match official class IDs")
            return mask.copy()
        if values.ndim == 2:
            return _validate_ids(values, str(path)).copy()
        if image.mode != "RGB" or values.shape[-1] != 3:
            raise ValueError(f"{path}: unsupported mask mode {image.mode}; expected class IDs or official RGB")
        packed = (values[:, :, 0].astype(np.uint32) << 16) | (values[:, :, 1].astype(np.uint32) << 8) | values[:, :, 2]
        official = (PALETTE[:, 0].astype(np.uint32) << 16) | (PALETTE[:, 1].astype(np.uint32) << 8) | PALETTE[:, 2]
        color_to_id = {int(color): class_id for class_id, color in enumerate(official)}
        unique, inverse = np.unique(packed, return_inverse=True)
        unknown = [int(color) for color in unique if int(color) not in color_to_id]
        if unknown:
            colors = [[color >> 16, (color >> 8) & 255, color & 255] for color in unknown[:10]]
            raise ValueError(f"{path}: unknown RGB mask colors {colors}; only exact official colors are accepted")
        ids = np.array([color_to_id[int(color)] for color in unique], dtype=np.uint8)
        return ids[inverse].reshape(packed.shape)


def validate_manifest(manifest: Any) -> dict[str, list[dict[str, str]]]:
    """Validate the explicit schema and canonical image basename identifiers."""
    if not isinstance(manifest, dict) or set(manifest) != set(SPLITS):
        raise ValueError("Manifest must contain exactly train, val, and test lists")
    for split in SPLITS:
        records = manifest[split]
        if not isinstance(records, list):
            raise ValueError(f"Manifest {split} must be a list")
        seen: set[str] = set()
        required = {"id", "image"} if split == "test" else {"id", "image", "mask"}
        for record in records:
            if not isinstance(record, dict) or set(record) != required:
                raise ValueError(f"Manifest {split} records must contain exactly {sorted(required)}")
            if any(not isinstance(value, str) or not value for value in record.values()):
                raise ValueError(f"Manifest {split} record values must be nonempty strings")
            identifier = record["id"]
            if identifier in {".", ".."} or "/" in identifier or "\\" in identifier or identifier.lower().endswith(".png"):
                raise ValueError(f"Invalid basename ID: {identifier!r}; use the image stem without .png")
            if identifier in seen:
                raise ValueError(f"Duplicate ID in {split}: {identifier}")
            seen.add(identifier)
            for key in required - {"id"}:
                if not Path(record[key]).is_absolute():
                    raise ValueError(f"{split}/{identifier}: {key} path must be absolute")
            if Path(record["image"]).stem != identifier:
                raise ValueError(f"{split}/{identifier}: ID must match image basename stem")
    return manifest


def read_manifest(path: str | Path) -> dict[str, list[dict[str, str]]]:
    with Path(path).open(encoding="utf-8") as source:
        return validate_manifest(json.load(source))


def prediction_set_sha256(directory: str | Path, records: list[dict[str, str]]) -> str:
    """Bind measured scores and timing to the exact named PNG content set."""
    hashes = {}
    for record in records:
        name = record["id"] + ".png"
        digest = hashlib.sha256()
        with (Path(directory) / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def data_input_sha256(manifest: dict, *, splits: tuple[str, ...] = ("train", "val")) -> str:
    """Fingerprint explicit authorized inputs, never discover hidden label files."""
    hashes = {}
    for split in splits:
        if split not in SPLITS:
            raise ValueError("Unknown split in data fingerprint")
        for record in manifest[split]:
            for key in (("image",) if split == "test" else ("image", "mask")):
                if key not in record:
                    continue
                path = str(Path(record[key]).resolve())
                if path in hashes:
                    continue
                digest = hashlib.sha256()
                with Path(path).open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                hashes[path] = digest.hexdigest()
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, allow_nan=False).encode()).hexdigest()


def build_manifest(
    root: str | Path,
    *,
    mask_dirs: dict[str, str | Path] | None = None,
    mask_template: str = "{stem}.png",
) -> dict[str, list[dict[str, str]]]:
    """Match image names explicitly; missing annotations never get guessed."""
    root = Path(root).expanduser().resolve(strict=True)
    mask_dirs = mask_dirs or {}
    if set(mask_dirs) - {"train", "val"}:
        raise ValueError("Mask directories may only be specified for train and val")
    manifest: dict[str, list[dict[str, str]]] = {}
    for split in SPLITS:
        image_dir = root / "images" / split
        if not image_dir.is_dir():
            raise ValueError(f"Missing image directory: {image_dir}")
        images = sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise ValueError(f"No images found in {image_dir}")
        mask_dir = Path(mask_dirs.get(split, root / "annotations" / "gt" / split)).expanduser()
        if split != "test" and not mask_dir.is_dir():
            raise ValueError(f"Missing mask directory: {mask_dir}; supply --mask-dir {split}=/absolute/path")
        records: list[dict[str, str]] = []
        matched_masks: set[Path] = set()
        for image in images:
            record = {"id": image.stem, "image": str(image.resolve(strict=True))}
            if split != "test":
                try:
                    mask_name = mask_template.format(stem=image.stem, name=image.name)
                except (KeyError, ValueError) as error:
                    raise ValueError("Mask template supports only {stem} and {name}") from error
                if not mask_name or Path(mask_name).name != mask_name or "\\" in mask_name:
                    raise ValueError("Mask template must produce one flat filename")
                mask = (mask_dir / mask_name).resolve()
                if not mask.is_file():
                    raise ValueError(f"Missing mask for {image.name}: {mask}; specify the exact --mask-template or --mask-suffix")
                if mask in matched_masks:
                    raise ValueError(f"Multiple images matched the same mask: {mask}")
                matched_masks.add(mask)
                record["mask"] = str(mask)
            records.append(record)
        manifest[split] = records
    return validate_manifest(manifest)


def audit_manifest(
    manifest: dict[str, list[dict[str, str]]],
    *,
    expected_size: tuple[int, int] = (1024, 512),
    strict_counts: bool = False,
) -> dict[str, Any]:
    """Audit decoded pixel hashes, class counts, shape, and split integrity.

    Returns all detected errors in a report. The CLI refuses to write a build
    manifest or exit successfully when the report contains integrity errors.
    expected_size is (width, height), matching Pillow conventions.
    """
    validate_manifest(manifest)
    if len(expected_size) != 2 or any(value <= 0 for value in expected_size):
        raise ValueError("Expected size must be positive (width, height)")
    errors: list[str] = []
    warnings: list[str] = []
    histograms: dict[str, list[int]] = {}
    hashes: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen_ids: dict[str, list[str]] = defaultdict(list)
    sizes: dict[str, list[list[int]]] = {}
    for split in SPLITS:
        histogram = np.zeros(NUM_CLASSES, dtype=np.int64)
        split_sizes: set[tuple[int, int]] = set()
        if strict_counts and len(manifest[split]) != OFFICIAL_COUNTS[split]:
            errors.append(f"{split}: expected {OFFICIAL_COUNTS[split]} images, got {len(manifest[split])}")
        for record in manifest[split]:
            identifier = record["id"]
            seen_ids[identifier].append(split)
            try:
                with Image.open(record["image"]) as image:
                    size = image.size
                    rgb = np.asarray(image.convert("RGB"))
                split_sizes.add(size)
                if size != expected_size:
                    errors.append(f"{split}/{identifier}: image size {size}, expected {expected_size}")
                digest = hashlib.sha256(str(rgb.shape).encode("ascii") + rgb.tobytes()).hexdigest()
                hashes[digest].append({"split": split, "id": identifier, "image": record["image"]})
                if split != "test":
                    mask = load_mask(record["mask"])
                    if mask.shape != (size[1], size[0]):
                        errors.append(f"{split}/{identifier}: mask shape {mask.shape} differs from image {(size[1], size[0])}")
                    histogram += np.bincount(mask.ravel(), minlength=NUM_CLASSES)
            except (OSError, ValueError) as error:
                errors.append(f"{split}/{identifier}: {error}")
        histograms[split] = histogram.tolist()
        sizes[split] = [list(size) for size in sorted(split_sizes)]
    duplicate_groups = [records for records in hashes.values() if len(records) > 1]
    for records in duplicate_groups:
        names = ", ".join(f"{record['split']}/{record['id']}" for record in records)
        if len({record["split"] for record in records}) > 1:
            errors.append(f"Duplicate decoded image across splits (leakage): {names}")
        else:
            warnings.append(f"Duplicate decoded image within split: {names}")
    for identifier, splits in seen_ids.items():
        if len(set(splits)) > 1:
            errors.append(f"Repeated image ID across splits (leakage): {identifier} in {', '.join(splits)}")
    return {
        "counts": {split: len(manifest[split]) for split in SPLITS},
        "expected_size": list(expected_size),
        "image_sizes": sizes,
        "class_names": CLASS_NAMES,
        "class_histograms": histograms,
        "duplicate_images": duplicate_groups,
        "hash_algorithm": "sha256(decoded RGB shape + pixels)",
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        json.dump(payload, destination, indent=2)
        destination.write("\n")


def main() -> None:
    from palmcity.storage import require_workspace, safe_output_path

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Build and audit an explicit manifest")
    build.add_argument("--root", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--mask-dir", action="append", default=[], metavar="SPLIT=/ABS/PATH")
    templates = build.add_mutually_exclusive_group()
    templates.add_argument("--mask-template", default="{stem}.png")
    templates.add_argument("--mask-suffix", help="For example _gt maps image.png to image_gt.png")
    audit = commands.add_parser("audit", help="Audit an existing manifest")
    audit.add_argument("--manifest", required=True)
    for command in (build, audit):
        command.add_argument("--report", required=True)
        command.add_argument("--strict-counts", action="store_true")
        command.add_argument("--width", type=int, default=1024)
        command.add_argument("--height", type=int, default=512)
    args = parser.parse_args()
    try:
        workspace = require_workspace()
        report_path = safe_output_path(args.report, workspace)
        if report_path.exists():
            raise ValueError(f"Refusing to overwrite report: {report_path}")
        if args.command == "build":
            output_path = safe_output_path(args.output, workspace)
            if output_path == report_path or output_path.exists():
                raise ValueError(f"Manifest output must be new and different from report: {output_path}")
            mask_dirs: dict[str, str] = {}
            for mapping in args.mask_dir:
                split, separator, directory = mapping.partition("=")
                if not separator or split not in {"train", "val"} or not Path(directory).is_absolute() or split in mask_dirs:
                    raise ValueError("Each --mask-dir must be unique train=/absolute/path or val=/absolute/path")
                mask_dirs[split] = directory
            template = "{stem}" + args.mask_suffix + ".png" if args.mask_suffix is not None else args.mask_template
            manifest = build_manifest(args.root, mask_dirs=mask_dirs, mask_template=template)
        else:
            manifest = read_manifest(args.manifest)
        report = audit_manifest(manifest, expected_size=(args.width, args.height), strict_counts=args.strict_counts)
        manifest_bytes = ((json.dumps(manifest, indent=2) + "\n").encode("utf-8")
                          if args.command == "build" else Path(args.manifest).read_bytes())
        report["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
        if report["valid"]:
            report["input_sha256"] = data_input_sha256(manifest)
            report["validation_input_sha256"] = data_input_sha256(manifest, splits=("val",))
        _write_json(report_path, report)
        if report["errors"]:
            raise ValueError(f"Audit failed with {len(report['errors'])} error(s); inspect {report_path}")
        if args.command == "build":
            _write_json(output_path, manifest)
            print(f"Manifest: {output_path}")
        print(f"Audit report: {report_path}; counts: {report['counts']}")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
