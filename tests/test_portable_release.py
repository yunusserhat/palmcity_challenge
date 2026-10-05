from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest

from palmcity import pretrained, storage
from palmcity.models import inference_model_config


@pytest.fixture
def portable_workspace(monkeypatch, tmp_path):
    workspace = tmp_path / "user-workspace"
    monkeypatch.setenv("PALMCITY_WORKSPACE", str(workspace))
    monkeypatch.setenv("PALMCITY_WORF_PROFILE", "0")
    # Synthetic nested workspaces can be long; reuse the script's validated
    # short temporary directory explicitly for these storage API tests.
    monkeypatch.setenv("PALMCITY_TMPDIR", os.environ["TMPDIR"])
    storage.prepare(0)
    return workspace


def test_explicit_workspace_prepare_and_confined_outputs(portable_workspace):
    assert storage.require_workspace() == portable_workspace
    assert storage.managed_directory(portable_workspace / "new" / "nested").is_dir()
    with pytest.raises(RuntimeError, match="below"):
        storage.managed_directory(portable_workspace.parent / "outside")


def test_workspace_refuses_home_root_and_symlink(monkeypatch, tmp_path):
    monkeypatch.setenv("PALMCITY_WORF_PROFILE", "0")
    monkeypatch.setenv("PALMCITY_WORKSPACE", str(Path.home()))
    with pytest.raises(RuntimeError, match="dedicated"):
        storage.configured_workspace()
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    monkeypatch.setenv("PALMCITY_WORKSPACE", str(link / "outputs"))
    with pytest.raises(RuntimeError, match="Symlink"):
        storage.configured_workspace()


def test_worf_profile_forbids_filesystem_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("PALMCITY_WORKSPACE", "/var/tmp/palmcity-synthetic-fallback-check")
    monkeypatch.setenv("PALMCITY_WORF_PROFILE", "1")
    monkeypatch.setattr(storage, "_scratch_mount", lambda: None)
    with pytest.raises(RuntimeError, match="below /scratch"):
        storage.configured_workspace()


def test_long_ipc_path_is_rejected_before_creation(monkeypatch, portable_workspace):
    directory = portable_workspace / ("long-temporary-name" * 8)
    monkeypatch.setenv("PALMCITY_TMPDIR", str(directory))
    with pytest.raises(RuntimeError, match="too long for AF_UNIX"):
        storage.temporary_directory(portable_workspace)
    assert not directory.exists()


def test_explicit_ipc_directory_is_owned_writable_and_on_workspace_filesystem(portable_workspace):
    directory = storage.temporary_directory(portable_workspace)
    assert directory.stat().st_uid == os.getuid()
    assert directory.stat().st_dev == portable_workspace.stat().st_dev
    assert (directory / "palmcity-write-access-check.txt").is_file()


def test_ipc_directory_rejects_different_filesystem(monkeypatch, portable_workspace):
    from types import SimpleNamespace

    directory = Path(os.environ["PALMCITY_TMPDIR"])
    original_stat = Path.stat

    def different_device(path, *args, **kwargs):
        actual = original_stat(path, *args, **kwargs)
        if path == directory and kwargs.get("follow_symlinks") is not False:
            return SimpleNamespace(st_uid=actual.st_uid, st_mode=actual.st_mode,
                                   st_dev=actual.st_dev + 1)
        return actual

    monkeypatch.setattr(Path, "stat", different_device)
    with pytest.raises(RuntimeError, match="same filesystem"):
        storage.temporary_directory(portable_workspace)


def test_pinned_snapshot_hashes_detect_modified_bytes(tmp_path):
    file = tmp_path / "model.safetensors"
    file.write_bytes(b"synthetic weights")
    source = {"files": [{"name": file.name, "bytes": file.stat().st_size,
                         "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}]}
    assert pretrained.verify_snapshot(tmp_path, source) == source["files"]
    file.write_bytes(b"Synthetic weights")
    with pytest.raises(ValueError, match="SHA-256"):
        pretrained.verify_snapshot(tmp_path, source)


def test_prepared_recipe_changes_only_initialization_path(tmp_path):
    root = Path(__file__).resolve().parents[1]
    recipe = json.loads((root / "configs/recipes/confirmation-eomt_dinov3_vit_large-seed123.json").read_text())
    source = next(s for s in pretrained.read_sources() if s["repo"] == recipe["pretrained_source"]["repo"])
    prepared = pretrained.prepare_config(recipe, source, tmp_path)
    expected = deepcopy(recipe)
    expected["model"]["pretrained_model_name_or_path"] = str(tmp_path)
    assert prepared == expected
    assert prepared["seed"] == 123
    assert prepared["max_optimizer_steps"] == 6351
    assert prepared["warmup_steps"] == 317
    assert prepared["validation_epochs"] == [6, 11, 16, 21, 26, 31, 36, 41, 46, 51]
    assert prepared["model"]["classes"] == 32
    different = deepcopy(source)
    different["revision"] = "0" * 40
    with pytest.raises(ValueError, match="differ"):
        pretrained.prepare_config(recipe, different, tmp_path)


def test_smp_pinned_encoder_is_removed_for_inference(tmp_path):
    from palmcity.experiments import _initialization

    source = {"repo": "smp-hub/resnet50.imagenet", "revision": "a" * 40}
    recipe = {"model": {"classes": 32, "encoder_weights": None}, "pretrained_source": source}
    prepared = pretrained.prepare_config(recipe, source, tmp_path)
    assert prepared["model"]["encoder_pretrained_path"] == str(tmp_path / "model.safetensors")
    assert _initialization(prepared) == "pretrained"
    assert inference_model_config(prepared["model"])["encoder_pretrained_path"] is None


def test_smp_initialization_uses_verified_local_tensors(monkeypatch, tmp_path):
    import segmentation_models_pytorch as smp
    import torch
    from safetensors.torch import save_file
    from torch import nn

    from palmcity.models import build_model

    class SyntheticModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(2, 1)

    def constructor(**kwargs):
        assert kwargs["encoder_weights"] is None
        return SyntheticModel()

    monkeypatch.setattr(smp, "DeepLabV3Plus", constructor)
    path = tmp_path / "model.safetensors"
    save_file({"weight": torch.tensor([[2.0, 3.0]]), "bias": torch.tensor([4.0])}, path)
    config = {"architecture": "deeplabv3plus", "classes": 32,
              "encoder_weights": None, "encoder_pretrained_path": str(path)}
    with pytest.raises(ValueError, match="explicit"):
        build_model(config)
    model = build_model(config, allow_pretrained_downloads=True)
    assert torch.equal(model.encoder.weight, torch.tensor([[2.0, 3.0]]))
    assert torch.equal(model.encoder.bias, torch.tensor([4.0]))


def test_published_recipes_have_no_machine_paths_and_fixed_budget():
    root = Path(__file__).resolve().parents[1]
    recipes = sorted((root / "configs/recipes").glob("*.json"))
    assert len(recipes) == 15
    for path in recipes:
        recipe = json.loads(path.read_text())
        assert recipe["batch_size"] * recipe["gradient_accumulation"] == 4
        assert recipe["image_size"] == [512, 1024]
        assert recipe["max_optimizer_steps"] > 0
        assert len(recipe["validation_epochs"]) == 10
        assert recipe["model"]["classes"] == 32
        assert "/scratch/" not in path.read_text()
        assert "/mnt/" not in path.read_text()
    assert len(pretrained.read_sources()) == 9
    diagnostics = sorted((root / "configs/diagnostics").glob("*.json"))
    assert len(diagnostics) == 4
    assert all(json.loads(p.read_text())["publication_selection_status"] ==
               "excluded_from_primary_selection" for p in diagnostics)
