"""Report label support and candidate neighboring frames; never read test labels."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

from palmcity.data import CLASS_NAMES, load_mask, read_manifest
from palmcity.download_data import write_json
from palmcity.storage import require_workspace, safe_output_path


def diagnose(manifest: dict) -> dict:
    result = {"label_support": {}, "filename_groups": {}, "neighboring_filename_candidates": [],
              "nearest_train_thumbnail_for_each_val": [],
              "limitations": [
                  "No location, route, timestamp or sequence metadata is supplied by the five authorized folders.",
                  "Numeric filename proximity and thumbnail similarity are candidate checks, not proof of shared location or leakage.",
                  "The official split is preserved; only train and val ground-truth masks are read."]}
    numeric_ids: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for split, records in manifest.items():
        support = np.zeros(32, dtype=np.int64)
        modes: Counter = Counter()
        names = Counter()
        for record in records:
            match = re.fullmatch(r"(.*?)(\d+)", record["id"])
            if match:
                prefix, number = match.groups()
                numeric_ids[prefix].append((int(number), split, record["id"]))
                names[prefix] += 1
            else:
                names["unparsed"] += 1
            if split != "test":
                mask = load_mask(record["mask"])
                support[np.unique(mask)] += 1
                with Image.open(record["mask"]) as image:
                    modes[image.mode] += 1
        result["filename_groups"][split] = dict(names)
        if split != "test":
            result["label_support"][split] = {
                "classes": [{"id": index, "name": name, "image_count": int(support[index])}
                            for index, name in enumerate(CLASS_NAMES)],
                "absent_classes": [name for index, name in enumerate(CLASS_NAMES) if support[index] == 0],
                "mask_modes": dict(modes)}
    for prefix, entries in sorted(numeric_ids.items()):
        entries.sort()
        for first, second in zip(entries, entries[1:]):
            if first[1] != second[1] and second[0] - first[0] <= 1:
                result["neighboring_filename_candidates"].append({
                    "prefix": prefix, "first": {"id": first[2], "split": first[1]},
                    "second": {"id": second[2], "split": second[1]}, "numeric_distance": second[0] - first[0]})
    result["neighbor_pairs_by_split_combination"] = dict(Counter(
        "/".join(sorted([pair["first"]["split"], pair["second"]["split"]]))
        for pair in result["neighboring_filename_candidates"]))

    def thumbnail(record: dict) -> np.ndarray:
        with Image.open(record["image"]) as image:
            return np.asarray(image.convert("RGB").resize((32, 16), Image.Resampling.BOX), dtype=np.float32).ravel()

    train = np.stack([thumbnail(record) for record in manifest["train"]])
    for record in manifest["val"]:
        distances = np.sqrt(np.mean((train - thumbnail(record)[None, :]) ** 2, axis=1))
        index = int(distances.argmin())
        result["nearest_train_thumbnail_for_each_val"].append({
            "val_id": record["id"], "train_id": manifest["train"][index]["id"],
            "thumbnail_rgb_rmse_0_255": float(distances[index])})
    result["nearest_train_thumbnail_for_each_val"].sort(key=lambda record: record["thumbnail_rgb_rmse_0_255"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report", default="reports/dataset-diagnostics.json")
    args = parser.parse_args()
    workspace = require_workspace()
    path = safe_output_path(args.report, workspace)
    if path.exists():
        parser.error("Diagnostics output must be new")
    result = diagnose(read_manifest(Path(args.manifest)))
    result["manifest"] = str(Path(args.manifest).resolve())
    write_json(path, result)
    print(json.dumps({"report": str(path), "label_support": result["label_support"],
                      "filename_neighbor_candidates": len(result["neighboring_filename_candidates"]),
                      "closest_thumbnail_pairs": result["nearest_train_thumbnail_for_each_val"][:5]}, indent=2))


if __name__ == "__main__":
    main()
