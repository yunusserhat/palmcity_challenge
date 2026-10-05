"""Validate and package flat PalmCity class-ID PNG submissions."""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import stat
import zipfile

import numpy as np
from PIL import Image

from palmcity.data import NUM_CLASSES, OFFICIAL_COUNTS, read_manifest, validate_manifest
from palmcity.storage import require_workspace, safe_output_path


def _expected_names(manifest: dict, allow_partial: bool = False) -> set[str]:
    validate_manifest(manifest)
    names = {record["id"] + ".png" for record in manifest["test"]}
    if not names:
        raise ValueError("Manifest test split is empty")
    if not allow_partial and len(names) != OFFICIAL_COUNTS["test"]:
        raise ValueError(f"Official submission requires {OFFICIAL_COUNTS['test']} test images, got {len(names)}; partial mode is for synthetic tests only")
    return names


def _check_names(actual: set[str], expected: set[str]) -> None:
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"Prediction names mismatch; missing={missing[:10]}, extra={extra[:10]}")


def _check_png(source: Path | io.BytesIO, label: str, expected_size: tuple[int, int]) -> None:
    with Image.open(source) as image:
        if image.format != "PNG":
            raise ValueError(f"{label}: expected PNG file")
        values = np.asarray(image)
        if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{label}: submission must contain single-channel class-ID PNG; RGB/colorized masks are invalid")
        if "transparency" in image.info:
            raise ValueError(f"{label}: transparent class-ID masks are invalid")
        if image.size != expected_size:
            raise ValueError(f"{label}: size {image.size}, expected {expected_size}")
        if np.any((values < 0) | (values >= NUM_CLASSES)):
            raise ValueError(f"{label}: invalid class IDs; expected 0..31")


def validate_predictions(
    predictions: str | Path,
    manifest: dict,
    *,
    expected_size: tuple[int, int] = (1024, 512),
    allow_partial: bool = False,
) -> list[Path]:
    """Reject missing/extra files and nested paths before making an archive."""
    predictions = Path(predictions)
    if not predictions.is_dir():
        raise ValueError(f"Prediction directory not found: {predictions}")
    entries = sorted(predictions.iterdir())
    if any(not entry.is_file() or entry.is_symlink() for entry in entries):
        raise ValueError("Prediction directory must contain only flat regular PNG files; no nested directories or symlinks")
    expected = _expected_names(manifest, allow_partial)
    _check_names({entry.name for entry in entries}, expected)
    for entry in entries:
        _check_png(entry, entry.name, expected_size)
    return entries


def validate_submission_zip(
    archive: str | Path,
    manifest: dict,
    *,
    expected_size: tuple[int, int] = (1024, 512),
    allow_partial: bool = False,
) -> None:
    """Validate every ZIP entry without extracting files or trusting paths."""
    expected = _expected_names(manifest, allow_partial)
    with zipfile.ZipFile(archive) as submission:
        entries = submission.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate paths in submission ZIP")
        if any(entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16) or Path(entry.filename).name != entry.filename or "\\" in entry.filename for entry in entries):
            raise ValueError("Submission ZIP must contain only flat PNG paths")
        _check_names(set(names), expected)
        for entry in entries:
            # Reject unexpectedly large decompressed entries before reading.
            if entry.file_size > max(16 * 1024 * 1024, expected_size[0] * expected_size[1] * 4):
                raise ValueError(f"{entry.filename}: PNG entry is unexpectedly large")
            _check_png(io.BytesIO(submission.read(entry)), entry.filename, expected_size)


def create_submission(
    manifest: dict,
    predictions: str | Path,
    output: str | Path,
    *,
    expected_size: tuple[int, int] = (1024, 512),
    allow_partial: bool = False,
) -> Path:
    """Create a new ZIP only inside the validated scratch workspace."""
    workspace = require_workspace()
    output = safe_output_path(output, workspace)
    if output.suffix.lower() != ".zip":
        raise ValueError("Submission output must have a .zip extension")
    if output.exists():
        raise ValueError(f"Refusing to overwrite submission: {output}")
    files = validate_predictions(predictions, manifest, expected_size=expected_size, allow_partial=allow_partial)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as destination:
        for path in files:
            destination.write(path, arcname=path.name)
    validate_submission_zip(output, manifest, expected_size=expected_size, allow_partial=allow_partial)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-partial", action="store_true", help="Allow a partial manifest for synthetic verification only")
    args = parser.parse_args()
    try:
        output = create_submission(read_manifest(args.manifest), args.predictions, args.output, allow_partial=args.allow_partial)
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Validated submission ZIP: {output}")


if __name__ == "__main__":
    main()
