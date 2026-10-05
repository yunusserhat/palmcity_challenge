"""Download only the five public, official PalmCity image/GT split folders."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

from PIL import Image

from palmcity.data import OFFICIAL_COUNTS
from palmcity.storage import managed_directory, require_free_space, require_workspace, safe_output_path


OFFICIAL_ROOT_ID = "1CKUzdQ8Tm74A6hHXWGd37NEI1JqTU7Jp"
FOLDER_URL = "https://drive.google.com/embeddedfolderview?id={}"
FILE_URL = "https://drive.usercontent.google.com/download?export=download&id={}"
ALLOWED_SPLITS = ("images/train", "images/val", "images/test", "annotations/gt/train", "annotations/gt/val")


class FolderParser(HTMLParser):
    """Read public Drive's complete embedded listing, including folders over 50 files."""

    def __init__(self) -> None:
        super().__init__()
        self.entries: list[dict[str, str]] = []
        self.href: str | None = None
        self.title_depth = 0
        self.title: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs = dict(attrs)
        if tag == "a":
            self.href = attrs.get("href")
            self.title = []
        if self.title_depth:
            self.title_depth += 1
        elif self.href and tag == "div" and attrs.get("class") == "flip-entry-title":
            self.title_depth = 1

    def handle_endtag(self, tag: str) -> None:
        if self.title_depth:
            self.title_depth -= 1
        if tag == "a" and self.href:
            match = re.fullmatch(r"https://drive\.google\.com/(?:file/d|drive/folders)/([-\w]+)(?:/view(?:\?.*)?)?", self.href)
            if match:
                name = "".join(self.title).strip()
                if not name or Path(name).name != name or "\\" in name or name in {".", ".."}:
                    raise ValueError("Unsafe or empty name in official Drive listing")
                self.entries.append({"id": match[1], "name": name,
                                     "kind": "folder" if "/drive/folders/" in self.href else "file"})
            self.href = None
            self.title_depth = 0

    def handle_data(self, data: str) -> None:
        if self.title_depth:
            self.title.append(data)


def parse_folder_listing(html: str) -> list[dict[str, str]]:
    parser = FolderParser()
    parser.feed(html)
    names = [entry["name"] for entry in parser.entries]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate names in official Drive listing")
    return parser.entries


def get_listing(folder_id: str) -> list[dict[str, str]]:
    with urllib.request.urlopen(FOLDER_URL.format(folder_id), timeout=60) as response:
        entries = parse_folder_listing(response.read().decode("utf-8"))
    if not entries:
        raise ValueError("Official Drive folder is empty or its public listing is unavailable")
    return entries


def discover_inventory() -> list[dict[str, str]]:
    listings: dict[str, list[dict[str, str]]] = {}

    def child(parent: str, name: str) -> str:
        if parent not in listings:
            listings[parent] = get_listing(parent)
        matches = [entry for entry in listings[parent] if entry["name"] == name and entry["kind"] == "folder"]
        if len(matches) != 1:
            raise ValueError(f"Missing or ambiguous authorized folder: {name}")
        return matches[0]["id"]

    inventory = []
    for relative in ALLOWED_SPLITS:
        folder_id = OFFICIAL_ROOT_ID
        for part in relative.split("/"):
            folder_id = child(folder_id, part)
        entries = get_listing(folder_id)
        if len(entries) != OFFICIAL_COUNTS[relative.rsplit("/", 1)[1]]:
            raise ValueError(f"{relative}: listing does not match the official split count")
        for entry in entries:
            if entry["kind"] != "file" or not entry["name"].lower().endswith(".png"):
                raise ValueError(f"Unexpected entry in authorized PNG split: {relative}")
            inventory.append({"id": entry["id"], "path": relative + "/" + entry["name"],
                              "folder_url": FOLDER_URL.format(folder_id),
                              "download_url": FILE_URL.format(entry["id"])})
    return sorted(inventory, key=lambda record: record["path"])


def inspect_png(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    with Image.open(path) as image:
        if image.format != "PNG" or image.size != (1024, 512):
            raise ValueError(f"Invalid official PNG dimensions/format: {path}")
        image.load()
        mode = image.mode
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size,
            "size": [1024, 512], "png_mode": mode}


def download_one(record: dict[str, str], root: Path, retries: int = 3) -> dict:
    destination = root / record["path"]
    if destination.is_symlink():
        raise ValueError(f"Symlink is not an authorized dataset file: {destination}")
    if destination.exists():
        return record | inspect_png(destination) | {"existing_file_verified": True}
    managed_directory(destination.parent)
    partial = destination.with_suffix(".png.download-partial")
    if partial.is_symlink():
        raise ValueError(f"Symlink is not an authorized partial download: {partial}")
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(record["download_url"], timeout=90) as response:
                if response.headers.get_content_type() not in {"image/png", "application/octet-stream"}:
                    raise ValueError(f"Drive returned a non-file response for {record['path']}")
                with partial.open("wb") as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
            info = inspect_png(partial)
            partial.replace(destination)
            return record | info | {"existing_file_verified": False}
        except (urllib.error.URLError, OSError, ValueError):
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 8))
    raise AssertionError("unreachable")


def write_json(path: Path, payload: dict) -> None:
    managed_directory(path.parent)
    temporary = path.with_suffix(path.suffix + ".write-partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/PalmCity")
    parser.add_argument("--provenance", default="reports/dataset-download.json")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")
    workspace = require_workspace()
    root = safe_output_path(args.root, workspace)
    report_path = safe_output_path(args.provenance, workspace)
    free_gib = require_free_space(workspace, 5)
    inventory = discover_inventory()
    for relative in ALLOWED_SPLITS:
        managed_directory(root / relative)
    report = {"source": "https://drive.google.com/drive/folders/" + OFFICIAL_ROOT_ID,
              "authorized_paths": list(ALLOWED_SPLITS), "hidden_test_labels": "never accessed",
              "listing_method": "complete public embeddedfolderview; exact counts checked",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(), "root": str(root),
              "free_gib_at_start": free_gib, "peak_data_preparation_estimate_gib": 5,
              "complete": False, "inventory": inventory, "files": []}
    write_json(report_path, report)
    started = time.perf_counter()
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, record, root): record for record in inventory}
        for future in as_completed(futures):
            record = futures[future]
            try:
                report["files"].append(future.result())
            except (OSError, ValueError) as error:
                errors.append({"path": record["path"], "error_type": type(error).__name__,
                               "message": str(error)})
            processed = len(report["files"]) + len(errors)
            if processed % 50 == 0 or processed == len(inventory):
                report["errors"] = errors
                write_json(report_path, report)
                print(f"Downloaded/verified {len(report['files'])}/{len(inventory)} PNGs; "
                      f"errors={len(errors)}; elapsed={time.perf_counter() - started:.1f}s", flush=True)
    report["files"].sort(key=lambda record: record["path"])
    report["errors"] = errors
    report["complete"] = not errors and len(report["files"]) == len(inventory)
    report["total_bytes"] = sum(record["bytes"] for record in report["files"])
    report["duration_seconds"] = time.perf_counter() - started
    write_json(report_path, report)
    if not report["complete"]:
        parser.exit(1, f"Incomplete download; restart safely with the same command. See {report_path}\n")
    print(f"Official dataset complete: {root}; provenance: {report_path}")


if __name__ == "__main__":
    main()
