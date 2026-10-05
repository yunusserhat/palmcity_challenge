"""Download-free Transformers segmentation adapters with PalmCity's unchanged IDs.

Constructors use local Python configs, never an online AutoConfig or processor.
Only explicit pretrained fields can reach ``from_pretrained``, behind the same
opt-in as SMP. The DINO linear head follows Meta's published linear probe design;
it is a baseline decoder, not the published 7B Mask2Former segmentor.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

NUM_CLASSES = 32


def _validate_semantic_targets(targets: torch.Tensor) -> None:
    if targets.ndim != 3 or targets.dtype not in (torch.int64, torch.int32, torch.uint8):
        raise ValueError("Semantic targets must be integer BHW tensors")
    if targets.numel() == 0 or targets.min() < 0 or targets.max() >= NUM_CLASSES:
        raise ValueError("Semantic targets must contain PalmCity IDs 0–31 without ignored labels")


def semantic_targets(targets: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """One binary mask per present class, including class 0 and Void=31."""
    _validate_semantic_targets(targets)
    labels = [target.unique(sorted=True).long() for target in targets]
    masks = [(target.unsqueeze(0) == label[:, None, None]).float()
             for target, label in zip(targets, labels, strict=True)]
    return masks, labels


def query_semantic_logits(
    class_logits: torch.Tensor, mask_logits: torch.Tensor, size: tuple[int, int],
) -> torch.Tensor:
    """Convert query masks to normalized 32-class probabilities for ensembles.

    The last query class is the separate no-object token, not PalmCity Void.
    This conversion keeps all 32 semantic classes and discards only no-object.
    Returning log probabilities lets the common inference softmax recover the
    probability mixture instead of exponentiating raw positive mask scores.
    """
    if class_logits.shape[-1] != NUM_CLASSES + 1:
        raise ValueError("Query classifier must contain 32 semantic classes plus no-object")
    masks = F.interpolate(mask_logits.float(), size=size, mode="bilinear", align_corners=False)
    classes = class_logits.float().softmax(dim=-1)[..., :NUM_CLASSES]
    scores = torch.einsum("bqc,bqhw->bchw", classes, masks.sigmoid()).clamp_min(1e-8)
    probabilities = scores / scores.sum(dim=1, keepdim=True)
    return probabilities.log()


def _variant_config(architecture: str, variant: str) -> dict[str, Any]:
    swin_large = {
        "model_type": "swin", "embed_dim": 192, "depths": [2, 2, 18, 2],
        "num_heads": [6, 12, 24, 48], "window_size": 12, "image_size": 384,
        "out_features": ["stage1", "stage2", "stage3", "stage4"],
    }
    if architecture == "segformer" and variant == "mit_b5":
        return {"depths": [3, 6, 40, 3], "hidden_sizes": [64, 128, 320, 512],
                "decoder_hidden_size": 768, "num_attention_heads": [1, 2, 5, 8]}
    if architecture == "upernet" and variant in {"convnext_large", "swin_large"}:
        backbone = swin_large if variant == "swin_large" else {
            "model_type": "convnext", "depths": [3, 3, 27, 3],
            "hidden_sizes": [192, 384, 768, 1536],
            "out_features": ["stage1", "stage2", "stage3", "stage4"],
        }
        return {"backbone_config": backbone, "hidden_size": 512,
                "auxiliary_in_channels": 768, "use_auxiliary_head": True}
    if architecture == "mask2former" and variant == "swin_large":
        return {"backbone_config": swin_large, "feature_size": 256,
                "mask_feature_size": 256, "hidden_dim": 256}
    if architecture in {"dinov3_linear", "eomt_dinov3"} and variant in {"vit_base", "vit_large"}:
        large = variant == "vit_large"
        values = {"hidden_size": 1024 if large else 768,
                  "intermediate_size": 4096 if large else 3072,
                  "num_hidden_layers": 24 if large else 12,
                  "num_attention_heads": 16 if large else 12,
                  "patch_size": 16, "num_register_tokens": 4,
                  "layer_norm_eps": 1e-6, "image_size": 512}
        if architecture == "eomt_dinov3":
            values.update({"num_queries": 100, "num_blocks": 4})
        return values
    if variant == "custom":
        return {}
    raise ValueError(f"Unsupported Transformers variant: {architecture}/{variant}")


def local_transformers_config(config: Mapping[str, Any]) -> Any:
    """Build a resolved config locally, overriding unsafe inherited label settings."""
    from transformers import (DINOv3ViTConfig, EomtDinov3Config, Mask2FormerConfig,
                              SegformerConfig, UperNetConfig)

    architecture = str(config["architecture"]).lower()
    constructors = {"segformer": SegformerConfig, "upernet": UperNetConfig,
                    "mask2former": Mask2FormerConfig, "dinov3_linear": DINOv3ViTConfig,
                    "eomt_dinov3": EomtDinov3Config}
    if architecture not in constructors:
        raise ValueError(f"Unsupported Transformers architecture: {architecture}")
    values = _variant_config(architecture, str(config.get("variant", "custom")))
    overrides = config.get("hf_config", {})
    if not isinstance(overrides, Mapping):
        raise ValueError("hf_config must be a local JSON object")
    values.update(copy.deepcopy(dict(overrides)))
    # Constructors do not load pretrained weights; prohibit recursive backbone
    # lookup options so an untrusted checkpoint cannot make a hidden Hub request.
    forbidden = {"backbone", "use_pretrained_backbone", "use_timm_backbone", "backbone_kwargs"}
    def reject_lookup(mapping: Mapping[str, Any]) -> None:
        for key, value in mapping.items():
            if key in forbidden and value not in (None, False, {}):
                raise ValueError(f"hf_config.{key} cannot perform an implicit backbone lookup")
            if isinstance(value, Mapping):
                reject_lookup(value)
    reject_lookup(values)
    from .data import CLASS_NAMES
    values.update({"num_labels": NUM_CLASSES, "id2label": dict(enumerate(CLASS_NAMES)),
                   "label2id": {label: i for i, label in enumerate(CLASS_NAMES)},
                   "semantic_loss_ignore_index": 255, "loss_ignore_index": 255,
                   "ignore_value": 255})
    result = constructors[architecture](**values)
    if hasattr(result, "use_pretrained_backbone"):
        result.use_pretrained_backbone = False
    return result


class DenseTransformerModel(nn.Module):
    input_multiple = 32
    uses_native_loss = False

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network
        self.backbone_module_prefixes = ("network.segformer.", "network.backbone.")

    def gradient_checkpointing_enable(self) -> None:
        _enable_checkpointing(self.network)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.network(pixel_values=images).logits
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)


class UperNetModel(DenseTransformerModel):
    """Use the library's CE and auxiliary supervision rather than drop the head."""
    uses_native_loss = True

    def training_loss(self, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _validate_semantic_targets(targets)  # no ADE label reduction or query-mask allocation
        return self.network(pixel_values=images, labels=targets).loss


class QueryTransformerModel(nn.Module):
    uses_native_loss = True

    def __init__(
        self, network: nn.Module, *, eomt: bool = False, attention_mask_schedule: Mapping | None = None,
    ) -> None:
        super().__init__()
        self.network = network
        self.eomt = eomt
        self.input_multiple = int(network.config.patch_size) if eomt else 32
        self.backbone_module_prefixes = (("network.embeddings.", "network.layers.", "network.layernorm.")
                                         if eomt else ("network.model.pixel_level_module.encoder.",))
        if eomt:
            blocks = int(network.config.num_blocks)
            default_start = [0.0] + [i / blocks for i in range(1, blocks)]
            default_end = [(i + 1) / blocks for i in range(blocks)]
            schedule = attention_mask_schedule or {}
            self.mask_start = list(schedule.get("start_fractions", default_start))
            self.mask_end = list(schedule.get("end_fractions", default_end))
            self.mask_power = float(schedule.get("power", 0.9))
            if (len(self.mask_start) != blocks or len(self.mask_end) != blocks
                    or any(not 0 <= start < end <= 1 for start, end in
                           zip(self.mask_start, self.mask_end, strict=True))
                    or not math.isfinite(self.mask_power) or self.mask_power <= 0):
                raise ValueError("EoMT attention mask schedule needs valid per-block fractions and power")
            self._training_attention_probs = [1.0] * blocks

    def train(self, mode: bool = True) -> QueryTransformerModel:
        super().train(mode)
        if self.eomt:
            probabilities = self._training_attention_probs if mode else [0.0] * len(self.mask_start)
            self.network.attn_mask_probs.copy_(self.network.attn_mask_probs.new_tensor(probabilities))
        return self

    def set_training_progress(self, completed_updates: int, total_updates: int) -> None:
        """Anneal EoMT's query attention masks on the saved optimizer-step budget.

        The polynomial mask schedule follows the official training code; start
        and end fractions are an explicit adaptation to a bounded PalmCity run.
        All accumulation microbatches in one optimizer update share the same
        probability. Validation/inference disables masks as the official recipe.
        """
        if not self.eomt:
            return
        if total_updates <= 0 or not 0 <= completed_updates <= total_updates:
            raise ValueError("Training progress must be within the positive optimizer-step budget")
        progress = completed_updates / total_updates
        self._training_attention_probs = [
            1.0 if progress < start else 0.0 if progress >= end
            else (1.0 - (progress - start) / (end - start)) ** self.mask_power
            for start, end in zip(self.mask_start, self.mask_end, strict=True)
        ]
        if self.training:
            self.network.attn_mask_probs.copy_(
                self.network.attn_mask_probs.new_tensor(self._training_attention_probs))

    def gradient_checkpointing_enable(self) -> None:
        _enable_checkpointing(self.network)

    def _prepare_grid(self, images: torch.Tensor) -> None:
        if self.eomt:
            patch = self.input_multiple
            if images.shape[-2] % patch or images.shape[-1] % patch:
                raise ValueError("EoMT input dimensions must be multiples of its patch size")
            # The pinned EoMT implementation initializes a square grid even
            # though DINOv3 RoPE accepts rectangles. Derive the head's reshape
            # grid from the actual panorama before every train/inference call.
            self.network.grid_size = (images.shape[-2] // patch, images.shape[-1] // patch)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self._prepare_grid(images)
        output = self.network(pixel_values=images)
        return query_semantic_logits(output.class_queries_logits, output.masks_queries_logits,
                                     images.shape[-2:])

    def training_loss(self, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        self._prepare_grid(images)
        masks, labels = semantic_targets(targets)
        output = self.network(pixel_values=images, mask_labels=masks, class_labels=labels)
        if output.loss is None:
            raise RuntimeError("Query model did not return its native Hungarian/mask loss")
        return output.loss


class DinoLinearModel(nn.Module):
    """Normalized last-layer patch features with BN/dropout/1x1 linear decoder."""
    uses_native_loss = False
    backbone_module_prefixes = ("backbone.",)

    def __init__(self, backbone: nn.Module, *, freeze_backbone: bool, dropout: float) -> None:
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.input_multiple = int(backbone.config.patch_size)
        width = int(backbone.config.hidden_size)
        self.decoder = nn.Sequential(nn.Dropout2d(dropout), nn.BatchNorm2d(width),
                                     nn.Conv2d(width, NUM_CLASSES, 1))
        nn.init.normal_(self.decoder[-1].weight, std=0.01)
        nn.init.zeros_(self.decoder[-1].bias)
        if freeze_backbone:
            self.backbone.requires_grad_(False)

    def gradient_checkpointing_enable(self) -> None:
        _enable_checkpointing(self.backbone)

    def train(self, mode: bool = True) -> DinoLinearModel:
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patch = self.input_multiple
        height, width = images.shape[-2:]
        if height % patch or width % patch:
            raise ValueError("DINOv3 input dimensions must be multiples of its patch size")
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_backbone):
            tokens = self.backbone(pixel_values=images).last_hidden_state
        prefix = 1 + int(self.backbone.config.num_register_tokens)
        patches = tokens[:, prefix:, :]
        features = patches.transpose(1, 2).reshape(images.shape[0], -1, height // patch, width // patch)
        logits = self.decoder(features)
        return F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)


def build_transformers_model(
    config: Mapping[str, Any], *, allow_pretrained_downloads: bool,
) -> nn.Module:
    from transformers import (DINOv3ViTModel, EomtDinov3ForUniversalSegmentation,
                              Mask2FormerForUniversalSegmentation,
                              SegformerForSemanticSegmentation, UperNetForSemanticSegmentation)

    architecture = str(config["architecture"]).lower()
    hf_config = local_transformers_config(config)
    constructors = {"segformer": SegformerForSemanticSegmentation,
                    "upernet": UperNetForSemanticSegmentation,
                    "mask2former": Mask2FormerForUniversalSegmentation,
                    "dinov3_linear": DINOv3ViTModel,
                    "eomt_dinov3": EomtDinov3ForUniversalSegmentation}
    pretrained = config.get("pretrained_model_name_or_path")
    pretrained_backbone = config.get("pretrained_backbone_name_or_path")
    if pretrained_backbone and architecture != "dinov3_linear":
        raise ValueError("Separate pretrained backbone loading is supported for dinov3_linear only")
    if pretrained and pretrained_backbone:
        raise ValueError("Choose exactly one pretrained initialization field")
    source = pretrained or pretrained_backbone
    if source and not allow_pretrained_downloads:
        raise ValueError("Pretrained initialization requires explicit --allow-pretrained-downloads")
    constructor = constructors[architecture]
    if source:
        network = _load_pretrained(constructor, source, hf_config, architecture)
    else:
        network = constructor(hf_config)
    if config.get("gradient_checkpointing", False):
        _enable_checkpointing(network)
    if architecture == "dinov3_linear":
        return DinoLinearModel(network, freeze_backbone=bool(config.get("freeze_backbone", False)),
                               dropout=float(config.get("decoder_dropout", 0.1)))
    if architecture in {"mask2former", "eomt_dinov3"}:
        return QueryTransformerModel(network, eomt=architecture == "eomt_dinov3",
                                     attention_mask_schedule=config.get("attention_mask_schedule"))
    if architecture == "upernet":
        return UperNetModel(network)
    return DenseTransformerModel(network)


def _load_pretrained(constructor: Any, source: str, hf_config: Any, architecture: str) -> nn.Module:
    from transformers import PretrainedConfig

    source_config, _ = PretrainedConfig.get_config_dict(source)
    if source_config.get("model_type") != hf_config.model_type:
        raise ValueError(f"Pretrained source model_type={source_config.get('model_type')!r} differs "
                         f"from {hf_config.model_type!r}; use a full compatible segmentation checkpoint")
    network, loading = constructor.from_pretrained(
        source, config=hf_config, ignore_mismatched_sizes=architecture != "dinov3_linear",
        output_loading_info=True,
    )
    permitted = {
        "segformer": {"decode_head.classifier.weight", "decode_head.classifier.bias"},
        "upernet": {"decode_head.classifier.weight", "decode_head.classifier.bias",
                    "auxiliary_head.classifier.weight", "auxiliary_head.classifier.bias"},
        "mask2former": {"class_predictor.weight", "class_predictor.bias", "criterion.empty_weight"},
        "eomt_dinov3": {"class_predictor.weight", "class_predictor.bias", "criterion.empty_weight"},
        "dinov3_linear": set(),
    }[architecture]
    mismatched = loading.get("mismatched_keys", [])
    changed = {item[0] if isinstance(item, (list, tuple)) else item for item in mismatched}
    changed.update(loading.get("missing_keys", []))
    unexpected = set(loading.get("unexpected_keys", []))
    if changed - permitted or unexpected or loading.get("error_msgs"):
        raise ValueError("Pretrained checkpoint has incompatible backbone/decoder tensors; "
                         f"missing/mismatched={sorted(changed - permitted)}, unexpected={sorted(unexpected)}")
    return network


def _enable_checkpointing(network: nn.Module) -> None:
    supported = [network] if getattr(network, "supports_gradient_checkpointing", False) else [
        module for module in network.modules()
        if module is not network and getattr(module, "supports_gradient_checkpointing", False)
    ]
    if not supported:
        raise ValueError(f"{type(network).__name__} does not support gradient checkpointing")
    for module in supported:
        module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
