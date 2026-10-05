"""Resolve immutable upstream snapshots and prepare portable training recipes."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re

from .storage import require_free_space, require_workspace, safe_output_path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def read_sources(path: str | Path | None = None) -> list[dict]:
    source_path = Path(path) if path else REPOSITORY_ROOT / "configs/pretrained-sources.json"
    sources = json.loads(source_path.read_text())
    if not isinstance(sources, list) or not sources:
        raise ValueError("Pretrained sources must be a nonempty list")
    for source in sources:
        if not re.fullmatch(r"[a-f0-9]{40}", source["revision"]):
            raise ValueError("Every pretrained revision must be an immutable 40-character commit")
        for item in source["files"]:
            if Path(item["name"]).name != item["name"] or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"]):
                raise ValueError("Snapshot files require safe names and SHA-256 digests")
    return sources


def verify_snapshot(snapshot: str | Path, source: dict) -> list[dict]:
    snapshot = Path(snapshot)
    verified = []
    for item in source["files"]:
        file = snapshot / item["name"]
        if not file.is_file() or file.stat().st_size != item["bytes"]:
            raise ValueError(f"Missing or wrong-size pinned snapshot file: {item['name']}")
        digest = hashlib.sha256()
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != item["sha256"]:
            raise ValueError(f"Pinned snapshot SHA-256 mismatch: {item['name']}")
        verified.append(dict(item))
    return verified


def prepare_config(config: dict, source: dict, snapshot: str | Path) -> dict:
    result = copy.deepcopy(config)
    recorded = result.get("pretrained_source", {})
    if (recorded.get("repo"), recorded.get("revision")) != (source["repo"], source["revision"]):
        raise ValueError("Recipe and pinned snapshot source differ")
    model = result["model"]
    if int(model.get("classes", 32)) != 32:
        raise ValueError("PalmCity recipes must preserve all 32 classes")
    if model.get("backend", "smp") == "smp":
        model["encoder_weights"] = None
        model["encoder_pretrained_path"] = str(Path(snapshot) / "model.safetensors")
    else:
        model["pretrained_model_name_or_path"] = str(Path(snapshot))
        model.pop("pretrained_backbone_name_or_path", None)
    result["initialization"] = "pretrained"
    return result


def _validate_cache(cache: Path) -> None:
    if not cache.is_absolute():
        raise ValueError("HF_HUB_CACHE must be absolute")
    if os.environ.get("PALMCITY_WORF_PROFILE") == "1" and not cache.is_dir():
        raise ValueError("The existing Worf cache must already exist")
    anchor = cache
    while not anchor.exists():
        anchor = anchor.parent
    if not anchor.is_dir() or anchor.stat().st_uid != os.getuid() or not os.access(anchor, os.W_OK | os.X_OK):
        raise ValueError("Hugging Face cache must have a user-owned, writable parent")


def prepare_recipe(recipe: str | Path, output: str | Path, *, download: bool = False,
                   sources_path: str | Path | None = None) -> Path:
    workspace = require_workspace()
    output = safe_output_path(output, workspace)
    provenance = output.with_suffix(".provenance.json")
    if output.exists() or provenance.exists():
        raise FileExistsError("Prepared config and provenance output must be new")
    recipe = Path(recipe)
    config = json.loads(recipe.read_text())
    recorded = config.get("pretrained_source", {})
    matches = [s for s in read_sources(sources_path) if s["repo"] == recorded.get("repo")]
    if len(matches) != 1:
        raise ValueError("Recipe needs exactly one known pretrained source")
    source = matches[0]
    cache = Path(os.environ.get("HF_HUB_CACHE", ""))
    _validate_cache(cache)
    cache_anchor = cache
    while not cache_anchor.exists():
        cache_anchor = cache_anchor.parent
    peak_gib = sum(item["bytes"] for item in source["files"]) * 2 / 1024**3 + 1
    require_free_space(cache_anchor, peak_gib if download else 0)
    if download and os.environ.get("HF_HUB_OFFLINE") == "1":
        raise ValueError("Explicit download requires HF_HUB_OFFLINE=0 before scripts/run.sh")
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(repo_id=source["repo"], revision=source["revision"],
                                 cache_dir=cache, local_files_only=not download,
                                 allow_patterns=[item["name"] for item in source["files"]])
    verified = verify_snapshot(snapshot, source)
    resolved = prepare_config(config, source, snapshot)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"recipe_sha256": hashlib.sha256(recipe.read_bytes()).hexdigest(),
               "source": source, "files_verified": verified,
               "prepared_config_sha256": hashlib.sha256((json.dumps(resolved, indent=2) + "\n").encode()).hexdigest(),
               "snapshot": str(snapshot), "cache": str(cache), "network_allowed": download,
               "cache_free_gib": require_free_space(cache_anchor, 0),
               "download_peak_estimate_gib": peak_gib}
    provenance.write_text(json.dumps(payload, indent=2) + "\n")
    output.write_text(json.dumps(resolved, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, help="Portable candidate or fixed-budget recipe JSON")
    parser.add_argument("--output", required=True, help="New prepared config inside PALMCITY_WORKSPACE")
    parser.add_argument("--sources", help="Override the pinned source metadata file")
    parser.add_argument("--download", action="store_true", help="Explicitly allow fetching only the pinned files")
    args = parser.parse_args()
    try:
        path = prepare_recipe(args.recipe, args.output, download=args.download, sources_path=args.sources)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, str(error) + "\n")
    print(path)


if __name__ == "__main__":
    main()
