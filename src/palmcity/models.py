"""Segmentation models and the preprocessing shared by training and prediction."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

NUM_CLASSES = 32
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)
PREPROCESSING = {"color": "RGB", "scale": 255.0, "mean": list(RGB_MEAN), "std": list(RGB_STD)}


class TinySegmentationModel(nn.Module):
    """A download-free model for pipeline smoke tests, not a competition baseline."""

    input_multiple = 1
    uses_native_loss = False

    def __init__(self, classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, classes, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)


def build_model(config: Mapping[str, Any], *, allow_pretrained_downloads: bool = False) -> nn.Module:
    """Construct an SMP or Transformers model with explicit pretrained opt-in."""
    classes = int(config.get("classes", NUM_CLASSES))
    if classes != NUM_CLASSES:
        raise ValueError("PalmCity checkpoints must contain all 32 classes, including Void=31")
    architecture = str(config.get("architecture", "deeplabv3plus")).lower()
    if architecture == "tiny":
        return TinySegmentationModel(classes)
    backend = str(config.get("backend", "smp")).lower()
    if backend == "transformers":
        if config.get("encoder_weights") is not None:
            raise ValueError("Transformers configs use pretrained_model_name_or_path, not encoder_weights")
        from .transformer_models import build_transformers_model
        return build_transformers_model(config, allow_pretrained_downloads=allow_pretrained_downloads)
    if backend != "smp":
        raise ValueError(f"Unsupported model backend: {backend}")
    architectures = {"deeplabv3plus": "DeepLabV3Plus", "unet": "Unet", "segformer": "Segformer"}
    if architecture not in architectures:
        raise ValueError(f"Unsupported model architecture: {architecture}")
    weights = config.get("encoder_weights")
    encoder_path = config.get("encoder_pretrained_path")
    if weights is not None and encoder_path:
        raise ValueError("Choose encoder_weights or encoder_pretrained_path, not both")
    if (weights is not None or encoder_path) and not allow_pretrained_downloads:
        raise ValueError("encoder_weights must be null unless --allow-pretrained-downloads is explicit")
    import segmentation_models_pytorch as smp

    constructor = getattr(smp, architectures[architecture], None)
    if constructor is None:
        raise RuntimeError(f"Installed segmentation-models-pytorch does not support {architecture}")
    kwargs = {
        "encoder_name": config.get("encoder_name", "resnet50"),
        "encoder_weights": weights,
        "in_channels": 3,
        "classes": classes,
        "activation": None,
    }
    permitted_options = {"encoder_depth", "decoder_channels", "encoder_output_stride"}
    for name in permitted_options:
        if name in config:
            kwargs[name] = config[name]
    model = constructor(**kwargs)
    if encoder_path:
        from pathlib import Path
        from safetensors.torch import load_file

        source = Path(encoder_path)
        if not source.is_file() or source.suffix != ".safetensors":
            raise ValueError("encoder_pretrained_path must be an existing safetensors file")
        model.encoder.load_state_dict(load_file(str(source), device="cpu"))
    model.input_multiple = 32
    model.uses_native_loss = False
    model.backbone_module_prefixes = ("encoder.",)
    return model


def inference_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct saved architecture without fetching initial pretrained weights."""
    result = copy.deepcopy(dict(config))
    for key in ("encoder_weights", "encoder_pretrained_path", "pretrained_model_name_or_path", "pretrained_backbone_name_or_path"):
        result[key] = None
    return result


def optimizer_parameter_groups(
    model: nn.Module, learning_rate: float, weight_decay: float, backbone_lr_multiplier: float = 1.0,
) -> list[dict[str, Any]]:
    """Explicit backbone/head rates, with biases and normalization left undecayed."""
    if learning_rate <= 0 or weight_decay < 0 or not 0 < backbone_lr_multiplier <= 1:
        raise ValueError("Optimizer rates/decay must be valid and backbone multiplier in (0,1]")
    prefixes = getattr(model, "backbone_module_prefixes", ())
    groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            key = (name.startswith(prefixes), parameter.ndim > 1 and not name.endswith(".bias"))
            groups.setdefault(key, []).append(parameter)
    return [{"params": parameters,
             "lr": learning_rate * (backbone_lr_multiplier if backbone else 1.0),
             "weight_decay": weight_decay if decay else 0.0,
             "group_name": f'{"backbone" if backbone else "head"}/{"decay" if decay else "no_decay"}'}
            for (backbone, decay), parameters in groups.items()]


def image_tensor(image: Image.Image, image_size: tuple[int, int]) -> torch.Tensor:
    """Resize the whole panorama and normalize RGB exactly as ImageNet encoders expect."""
    height, width = image_size
    if height <= 0 or width != 2 * height:
        raise ValueError("image_size must be [height, width] with a 2:1 panorama aspect ratio")
    if image.width * height != image.height * width:
        raise ValueError(f"Image {image.size} does not have the expected 2:1 panorama aspect ratio")
    image = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.asarray(RGB_MEAN, dtype=np.float32)) / np.asarray(RGB_STD, dtype=np.float32)
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def select_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type not in {"cuda", "cpu"}:
        raise ValueError("Use an explicit cuda:N device or cpu")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu explicitly for a synthetic smoke test")
        torch.cuda.set_device(device)
    return device


def amp_settings(device: torch.device, enabled: bool) -> tuple[bool, torch.dtype]:
    use_amp = enabled and device.type == "cuda"
    dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    return use_amp, dtype
