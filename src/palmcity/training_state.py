"""Small, tensor-safe state and provenance helpers for resumable experiments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def code_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    paths = sorted(root.glob("src/palmcity/*.py")) + sorted(root.glob("scripts/*.sh"))
    paths += [root / "pyproject.toml", root / "uv.lock"]
    hashes = {str(path.relative_to(root)): file_sha256(path) for path in paths if path.is_file()}
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False,
    )
    return {"sha256": json_sha256(hashes), "files": hashes,
            "git_revision": revision.stdout.strip() if revision.returncode == 0 else None}


def dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": sys.version.split()[0], "cuda": torch.version.cuda}
    for name in ("torch", "torchvision", "segmentation-models-pytorch", "numpy", "pillow", "transformers", "timm", "scipy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def data_identity(manifest: dict[str, Any], manifest_path: str | Path) -> dict[str, Any]:
    # Hash only authorized training/validation inputs; hidden test labels are never read.
    files: dict[str, str] = {}
    for split in ("train", "val"):
        for record in manifest.get(split, []):
            for key in ("image", "mask"):
                path = str(Path(record[key]).resolve())
                if path not in files:
                    files[path] = file_sha256(path)
    return {"manifest_sha256": file_sha256(manifest_path), "input_sha256": json_sha256(files),
            "input_files": files, "counts": {key: len(value) for key, value in manifest.items()}}


def capture_rng(generator: torch.Generator, device: torch.device) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {"algorithm": numpy_state[0], "keys": numpy_state[1].tolist(),
                  "position": numpy_state[2], "has_gauss": numpy_state[3], "cached_gaussian": numpy_state[4]},
        "torch": torch.get_rng_state(), "loader": generator.get_state(),
        "cuda": torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None,
    }


def restore_rng(state: dict[str, Any], generator: torch.Generator, device: torch.device) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["algorithm"], np.asarray(numpy_state["keys"], dtype=np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"]))
    torch.set_rng_state(state["torch"].cpu())
    generator.set_state(state["loader"].cpu())
    if device.type == "cuda":
        if state["cuda"] is None:
            raise ValueError("A CUDA resume needs the original CUDA RNG state")
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def cpu_state(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    return value


def atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_metrics(path: Path, history: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        for metrics in history:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
