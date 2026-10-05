"""Validate an explicitly selected, user-owned workspace before writing artifacts."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def _no_symlinks(path: Path) -> None:
    for parent in [*reversed(path.parents), path]:
        if parent.is_symlink():
            raise RuntimeError(f"Symlink not allowed in managed path: {parent}")


def _scratch_mount() -> None:
    """Keep the Worf no-fallback rule when its existing cache profile is active."""
    mounted = subprocess.run(["mountpoint", "-q", "/scratch"], check=False)
    if mounted.returncode or os.stat("/scratch").st_dev == os.stat("/").st_dev:
        raise RuntimeError("The separate /scratch filesystem is unavailable; no fallback allowed.")


def configured_workspace() -> Path:
    configured = os.environ.get("PALMCITY_WORKSPACE")
    if not configured or not Path(configured).is_absolute():
        raise RuntimeError("Run through scripts/run.sh with an absolute PALMCITY_WORKSPACE.")
    workspace = Path(configured).expanduser()
    if workspace == Path("/") or workspace == Path.home():
        raise RuntimeError("Choose a dedicated workspace directory, not a filesystem or home root.")
    _no_symlinks(workspace)
    if os.environ.get("PALMCITY_WORF_PROFILE") == "1":
        _scratch_mount()
        if not workspace.is_relative_to(Path("/scratch")):
            raise RuntimeError("The Worf workspace must be below /scratch; no fallback allowed.")
    return workspace


def _check_directory(path: Path) -> None:
    if not path.is_dir() or path.stat().st_uid != os.getuid():
        raise RuntimeError(f"Managed directory must be owned by this user: {path}")
    if not os.access(path, os.W_OK | os.X_OK):
        raise RuntimeError(f"Managed directory is not writable: {path}")


def require_workspace() -> Path:
    workspace = configured_workspace()
    _check_directory(workspace)
    if os.environ.get("PALMCITY_WORF_PROFILE") == "1":
        if workspace.stat().st_dev != os.stat("/scratch").st_dev:
            raise RuntimeError("The Worf workspace must be on the scratch filesystem.")
    return workspace


def managed_directory(path: Path) -> Path:
    """Create only directories inside the explicitly validated workspace."""
    workspace = require_workspace()
    path = Path(path).absolute()
    if not path.is_relative_to(workspace):
        raise RuntimeError(f"Managed directory must be below {workspace}: {path}")
    _no_symlinks(path)
    current = workspace
    for part in path.relative_to(workspace).parts:
        current /= part
        if not current.exists():
            current.mkdir(mode=0o700)
        _check_directory(current)
        if current.stat().st_dev != workspace.stat().st_dev:
            raise RuntimeError(f"Managed directory is on another filesystem: {current}")
    return path


def safe_output_path(path: str | Path, workspace: Path) -> Path:
    workspace = Path(workspace)
    if workspace != require_workspace():
        raise RuntimeError("Output workspace does not match the validated workspace.")
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    _no_symlinks(candidate)
    candidate = candidate.resolve()
    if not candidate.is_relative_to(workspace) or candidate == workspace:
        raise RuntimeError(f"Outputs must be inside {workspace}: {candidate}")
    for parent in [*candidate.parents, candidate]:
        if parent.is_relative_to(workspace) and parent.exists():
            if parent.stat().st_uid != os.getuid():
                raise RuntimeError(f"Output path is not owned by this user: {parent}")
            if parent.stat().st_dev != workspace.stat().st_dev:
                raise RuntimeError(f"Output path is on another filesystem: {parent}")
    return candidate


def require_free_space(path: Path, gib: float = 1.0) -> float:
    if gib < 0:
        raise ValueError("Required free space must be nonnegative.")
    free_gib = shutil.disk_usage(path).free / 1024**3
    if free_gib < gib:
        raise RuntimeError(f"Need {gib:.1f} GiB free; {free_gib:.1f} GiB available on {path}.")
    return free_gib


def temporary_directory(workspace: Path) -> Path:
    """Allow a short external IPC path only through an explicit user setting."""
    configured = os.environ.get("PALMCITY_TMPDIR")
    directory = Path(configured) if configured else workspace / "tmp"
    if not directory.is_absolute():
        raise RuntimeError("PALMCITY_TMPDIR must be an absolute path.")
    # Python appends /pymp-XXXXXXXX/listener-XXXXXXXX. Linux AF_UNIX paths
    # allow only 107 bytes; leave additional room for other IPC users.
    if len(os.fsencode(directory)) > 70:
        raise RuntimeError(
            "Temporary directory path is too long for AF_UNIX sockets. "
            "Set PALMCITY_TMPDIR to an absolute, short, user-owned directory "
            "on the workspace filesystem (at most 70 path bytes)."
        )
    if directory == Path("/") or directory == Path.home():
        raise RuntimeError("Choose a dedicated temporary directory, not a filesystem or home root.")
    _no_symlinks(directory)
    anchor = directory
    missing = []
    while not anchor.exists():
        missing.append(anchor)
        anchor = anchor.parent
    _check_directory(anchor)
    if anchor.stat().st_dev != workspace.stat().st_dev:
        raise RuntimeError("Temporary directory must be on the same filesystem as the workspace.")
    if os.environ.get("PALMCITY_WORF_PROFILE") == "1" and not directory.is_relative_to(Path("/scratch")):
        raise RuntimeError("The Worf temporary directory must remain on scratch.")
    require_free_space(anchor)
    for path in reversed(missing):
        path.mkdir(mode=0o700)
        _check_directory(path)
    _check_directory(directory)
    probe = directory / "palmcity-write-access-check.txt"
    _no_symlinks(probe)
    if probe.exists() and (not probe.is_file() or probe.stat().st_uid != os.getuid()):
        raise RuntimeError("Temporary write-access probe must be an owned regular file.")
    with probe.open("a") as handle:
        handle.write("Temporary directory preflight write succeeded.\n")
    return directory


def prepare(min_free_gib: float) -> dict:
    workspace = configured_workspace()
    nearest = workspace
    while not nearest.exists():
        nearest = nearest.parent
    _check_directory(nearest)
    free_gib = require_free_space(nearest, min_free_gib)
    missing = []
    current = workspace
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir(mode=0o700)
    require_workspace()
    temp_directory = temporary_directory(workspace)
    for name in (
        "data", "outputs", "manifests", "tmp", "reports", "verification", "pycache", "configs",
        "cache/uv", "cache/pip", "cache/torch", "cache/xdg",
    ):
        managed_directory(workspace / name)
    probe = workspace / "verification" / "write-access-check.txt"
    safe_output_path(probe, workspace)
    if probe.exists() and not probe.is_file():
        raise RuntimeError("Write-access check must be a regular file.")
    with probe.open("a") as handle:
        handle.write("Workspace preflight write succeeded.\n")
    return {
        "workspace": str(workspace), "free_gib": round(free_gib, 2),
        "filesystem_device": workspace.stat().st_dev, "owner_uid": workspace.stat().st_uid,
        "minimum_free_gib": min_free_gib, "dependency_install_peak_estimate_gib": 20,
        "dataset_preparation_peak_estimate_gib": 5,
        "worf_profile": os.environ.get("PALMCITY_WORF_PROFILE") == "1",
        "temporary_directory": str(temp_directory),
        "external_temporary_directory_explicitly_selected": bool(os.environ.get("PALMCITY_TMPDIR")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--temporary-directory", action="store_true")
    parser.add_argument("--min-free-gib", type=float, default=1.0)
    args = parser.parse_args()
    try:
        if args.temporary_directory:
            print(temporary_directory(require_workspace()))
            return
        if args.prepare:
            report = prepare(args.min_free_gib)
        else:
            workspace = require_workspace()
            report = {"workspace": str(workspace), "free_gib": require_free_space(workspace, args.min_free_gib)}
    except (RuntimeError, ValueError, OSError) as error:
        parser.exit(1, f"Workspace preflight failed: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
