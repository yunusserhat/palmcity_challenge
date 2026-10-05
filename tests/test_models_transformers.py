"""Real offline adapter forward/backward tests using tiny local random configs."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from palmcity.models import build_model, inference_model_config, optimizer_parameter_groups
from palmcity.transformer_models import query_semantic_logits, semantic_targets


def small_model_config(architecture: str, backbone: str = "swin") -> dict:
    config = {"backend": "transformers", "architecture": architecture, "variant": "custom",
              "classes": 32, "pretrained_model_name_or_path": None}
    hierarchical = {"model_type": "swin", "embed_dim": 8, "depths": [1, 1, 1, 1],
                    "num_heads": [1, 2, 4, 8], "window_size": 2,
                    "out_features": ["stage1", "stage2", "stage3", "stage4"]}
    if backbone == "convnext":
        hierarchical = {"model_type": "convnext", "depths": [1, 1, 1, 1],
                        "hidden_sizes": [8, 16, 32, 64],
                        "out_features": ["stage1", "stage2", "stage3", "stage4"]}
    if architecture == "segformer":
        values = {"depths": [1, 1, 1, 1], "hidden_sizes": [8, 16, 32, 64],
                  "num_attention_heads": [1, 2, 4, 8], "sr_ratios": [4, 2, 1, 1],
                  "decoder_hidden_size": 16}
    elif architecture == "upernet":
        values = {"backbone_config": hierarchical, "hidden_size": 16,
                  "auxiliary_in_channels": 32, "auxiliary_channels": 8, "pool_scales": [1, 2]}
    elif architecture == "mask2former":
        values = {"backbone_config": hierarchical, "feature_size": 32, "mask_feature_size": 32,
                  "hidden_dim": 32, "encoder_feedforward_dim": 64, "encoder_layers": 1,
                  "decoder_layers": 2, "num_attention_heads": 4, "dim_feedforward": 64,
                  "train_num_points": 16, "num_queries": 4}
    else:
        values = {"hidden_size": 32, "intermediate_size": 64, "num_attention_heads": 4,
                  "num_hidden_layers": 2, "num_register_tokens": 4, "patch_size": 8,
                  "image_size": 64}
        if architecture == "eomt_dinov3":
            values.update(num_queries=4, num_blocks=1, train_num_points=16, num_upscale_blocks=1)
    config["hf_config"] = values
    return config


@pytest.fixture(autouse=True)
def offline_and_small_threads(monkeypatch: pytest.MonkeyPatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Random model integration must not access the network")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("architecture,backbone", [
    ("segformer", "swin"), ("upernet", "swin"), ("upernet", "convnext"),
    ("mask2former", "swin"), ("dinov3_linear", "swin"), ("eomt_dinov3", "swin"),
])
def test_offline_real_forward_backward_and_checkpoint_roundtrip(
    architecture: str, backbone: str, tmp_path: Path,
) -> None:
    torch.manual_seed(7)
    config = small_model_config(architecture, backbone)
    model = build_model(config)
    images = torch.randn(2, 3, 64, 128)
    targets = torch.zeros(2, 64, 128, dtype=torch.long)
    targets[:, :, 64:] = 31
    model.train()
    logits = model(images)
    assert logits.shape == (2, 32, 64, 128)
    assert torch.isfinite(logits).all()
    loss = (model.training_loss(images, targets) if model.uses_native_loss
            else F.cross_entropy(logits, targets))
    assert loss.ndim == 0 and torch.isfinite(loss) and loss > 0
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in model.parameters())
    model.eval()
    with torch.no_grad():
        expected = model(images)
    path = tmp_path / "random.pt"
    torch.save(model.state_dict(), path)
    restored = build_model(inference_model_config(config)).eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(images), expected)


def test_query_targets_preserve_zero_and_void_and_reject_ignore() -> None:
    targets = torch.tensor([[[0, 31], [31, 0]]])
    masks, labels = semantic_targets(targets)
    assert labels[0].tolist() == [0, 31]
    torch.testing.assert_close(masks[0].sum(dim=0), torch.ones(2, 2))
    with pytest.raises(ValueError, match="0–31"):
        semantic_targets(torch.tensor([[[255]]]))


def test_query_probability_conversion_drops_only_no_object() -> None:
    classes = torch.full((1, 2, 33), -20.0)
    classes[0, 0, 31] = 20
    classes[0, 1, 32] = 20  # no-object must not become Void
    masks = torch.full((1, 2, 2, 4), 5.0)
    logits = query_semantic_logits(classes, masks, (4, 8))
    probabilities = logits.softmax(dim=1)
    assert torch.all(probabilities.argmax(dim=1) == 31)
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(1, 4, 8))


def test_pretrained_and_implicit_lookup_rejected_before_network() -> None:
    config = small_model_config("segformer")
    config["pretrained_model_name_or_path"] = "nvidia/segformer-b5-finetuned-ade-640-640"
    with pytest.raises(ValueError, match="explicit"):
        build_model(config)
    config["pretrained_model_name_or_path"] = None
    config["hf_config"]["backbone"] = "arbitrary-online-model"
    with pytest.raises(ValueError, match="implicit"):
        build_model(config)
    config["hf_config"] = {"backbone_config": {"use_pretrained_backbone": True}}
    with pytest.raises(ValueError, match="implicit"):
        build_model(config)


def test_inference_initialization_flags_are_cleared_without_mutation() -> None:
    config = {"encoder_weights": "imagenet", "pretrained_model_name_or_path": "online",
              "pretrained_backbone_name_or_path": "online-backbone", "hf_config": {"depths": [1]}}
    sanitized = inference_model_config(config)
    assert all(sanitized[key] is None for key in ("encoder_weights", "pretrained_model_name_or_path",
                                                "pretrained_backbone_name_or_path"))
    sanitized["hf_config"]["depths"].append(2)
    assert config["hf_config"]["depths"] == [1]


def test_dino_patch_grid_freeze_checkpointing_and_optimizer_groups() -> None:
    config = small_model_config("dinov3_linear")
    model = build_model(config)
    model.gradient_checkpointing_enable()
    groups = optimizer_parameter_groups(model, 0.001, 0.01, 0.1)
    assert {group["lr"] for group in groups} == {0.001, 0.0001}
    assert sum(len(group["params"]) for group in groups) == len(list(model.parameters()))
    with pytest.raises(ValueError, match="multiples"):
        model(torch.randn(2, 3, 33, 64))
    config["freeze_backbone"] = True
    frozen = build_model(config).train()
    assert not frozen.backbone.training
    assert not any(parameter.requires_grad for parameter in frozen.backbone.parameters())
    frozen(torch.randn(2, 3, 32, 64)).sum().backward()
    assert all(parameter.grad is None for parameter in frozen.backbone.parameters())


@pytest.mark.parametrize("architecture", ["segformer", "upernet", "mask2former", "eomt_dinov3"])
def test_local_synthetic_pretrained_changes_only_semantic_heads(
    architecture: str, tmp_path: Path,
) -> None:
    config = small_model_config(architecture)
    random = build_model(config)
    source_config = type(random.network.config).from_dict(random.network.config.to_dict())
    source_config.num_labels = 3
    source_network = type(random.network)(source_config)
    source = tmp_path / "synthetic-hf-source"
    source_network.save_pretrained(source)
    config["pretrained_model_name_or_path"] = str(source)
    loaded = build_model(config, allow_pretrained_downloads=True)
    assert loaded.network.config.num_labels == 32
    if architecture == "segformer":
        source_backbone, target_backbone = source_network.segformer, loaded.network.segformer
    elif architecture == "upernet":
        source_backbone, target_backbone = source_network.backbone, loaded.network.backbone
    elif architecture == "mask2former":
        source_backbone = source_network.model.pixel_level_module.encoder
        target_backbone = loaded.network.model.pixel_level_module.encoder
    else:
        source_backbone, target_backbone = source_network.embeddings, loaded.network.embeddings
    for expected, actual in zip(source_backbone.parameters(), target_backbone.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)


def test_plain_dino_checkpoint_cannot_silently_initialize_eomt(tmp_path: Path) -> None:
    source_config = small_model_config("dinov3_linear")
    source_network = build_model(source_config).backbone
    source = tmp_path / "synthetic-plain-dino"
    source_network.save_pretrained(source)
    config = small_model_config("eomt_dinov3")
    config["pretrained_model_name_or_path"] = str(source)
    with pytest.raises(ValueError, match="model_type"):
        build_model(config, allow_pretrained_downloads=True)


def test_pretrained_architecture_mismatch_is_rejected(tmp_path: Path) -> None:
    config = small_model_config("segformer")
    source = tmp_path / "synthetic-incompatible-depth"
    build_model(config).network.save_pretrained(source)
    config["pretrained_model_name_or_path"] = str(source)
    config["hf_config"]["depths"] = [2, 1, 1, 1]
    with pytest.raises(ValueError, match="incompatible backbone"):
        build_model(config, allow_pretrained_downloads=True)


def test_eomt_attention_schedule_restores_train_probability_after_eval() -> None:
    config = small_model_config("eomt_dinov3")
    model = build_model(config)
    model.set_training_progress(5, 10)
    expected = 0.5**0.9
    torch.testing.assert_close(model.network.attn_mask_probs, torch.tensor([expected]))
    model.eval()
    assert model.network.attn_mask_probs.item() == 0
    model.train()
    torch.testing.assert_close(model.network.attn_mask_probs, torch.tensor([expected]))
    model.set_training_progress(10, 10)
    assert model.network.attn_mask_probs.item() == 0
    with pytest.raises(ValueError, match="budget"):
        model.set_training_progress(0, 0)
