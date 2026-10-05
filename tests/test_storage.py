from pathlib import Path

import pytest

from palmcity import storage


def test_workspace_requires_explicit_environment(monkeypatch):
    monkeypatch.delenv("PALMCITY_WORKSPACE", raising=False)
    with pytest.raises(RuntimeError, match="scripts/run.sh"):
        storage.require_workspace()


def test_output_cannot_escape_workspace(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "require_workspace", lambda: tmp_path)
    with pytest.raises(RuntimeError, match="Outputs must be inside"):
        storage.safe_output_path("../home-output.pt", tmp_path)
    with pytest.raises(RuntimeError, match="Outputs must be inside"):
        storage.safe_output_path(Path("/tmp/home-output.pt"), tmp_path)


def test_output_symlink_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "require_workspace", lambda: tmp_path)
    target = tmp_path / "actual"
    target.mkdir()
    link = tmp_path / "symlink"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="Symlink"):
        storage.safe_output_path(link / "model.pt", tmp_path)


def test_valid_relative_output_stays_in_workspace(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "require_workspace", lambda: tmp_path)
    assert storage.safe_output_path("outputs/run/model.pt", tmp_path) == tmp_path / "outputs/run/model.pt"


def test_unmounted_scratch_is_rejected(monkeypatch):
    class Unmounted:
        returncode = 1

    monkeypatch.setattr(storage.subprocess, "run", lambda *args, **kwargs: Unmounted())
    with pytest.raises(RuntimeError, match="no fallback"):
        storage._scratch_mount()


def test_insufficient_space_is_rejected(monkeypatch, tmp_path):
    class Usage:
        free = 512

    monkeypatch.setattr(storage.shutil, "disk_usage", lambda _: Usage())
    with pytest.raises(RuntimeError, match="Need 20.0 GiB"):
        storage.require_free_space(tmp_path, 20)
