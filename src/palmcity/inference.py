"""Image-at-a-time probability inference with bounded panorama TTA and tiling."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .models import NUM_CLASSES


@dataclass(frozen=True)
class InferenceOptions:
    """Scales refer to each checkpoint's configured panorama input size."""

    scales: tuple[float, ...] = (1.0,)
    window_size: tuple[int, int] | None = None
    overlap: float = 0.5
    max_scaled_pixels: int = 2_097_152
    max_tiles: int = 512

    def __post_init__(self) -> None:
        if not 1 <= len(self.scales) <= 8 or not all(
            math.isfinite(scale) and scale > 0 for scale in self.scales
        ):
            raise ValueError("Use 1 to 8 finite, positive inference scales")
        if not math.isfinite(self.overlap) or not 0 <= self.overlap < 1:
            raise ValueError("Window overlap must be finite and in [0, 1)")
        if self.window_size is not None:
            height, width = self.window_size
            if not all(isinstance(value, int) and value > 0 for value in (height, width)) or width != 2 * height:
                raise ValueError("window_size must be positive [height, width] with 2:1 aspect ratio")
        for name in ("max_scaled_pixels", "max_tiles"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")


def _starts(length: int, window: int, overlap: float) -> list[int]:
    if length <= window:
        return [0]
    stride = max(1, int(window * (1 - overlap)))
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def inference_plans(
    base_size: tuple[int, int], options: InferenceOptions,
) -> list[dict]:
    """Validate all resource bounds before any forward or output creation."""
    height, width = base_size
    if height <= 0 or width != 2 * height:
        raise ValueError("Inference input must preserve the 2:1 panorama aspect ratio")
    plans = []
    for scale in options.scales:
        if not math.isfinite(height * scale):
            raise ValueError("Inference scale exceeds max_scaled_pixels")
        scaled_height = max(1, int(math.floor(height * scale + 0.5)))
        scaled_width = 2 * scaled_height
        if scaled_height * scaled_width > options.max_scaled_pixels:
            raise ValueError("Inference scale exceeds max_scaled_pixels; reduce scales or explicitly raise the bound")
        window = options.window_size or (scaled_height, scaled_width)
        ys = _starts(scaled_height, window[0], options.overlap)
        xs = _starts(scaled_width, window[1], options.overlap)
        if len(ys) * len(xs) > options.max_tiles:
            raise ValueError("Sliding inference exceeds max_tiles; increase window size or reduce overlap")
        plans.append({"scale": scale, "size": [scaled_height, scaled_width],
                      "tiles": len(ys) * len(xs), "y_starts": ys, "x_starts": xs})
    return plans


def _validate_logits(logits: torch.Tensor, batch_size: int) -> None:
    if not isinstance(logits, torch.Tensor) or logits.ndim != 4 or logits.shape[0] != batch_size or logits.shape[1] != NUM_CLASSES:
        raise RuntimeError("Prediction model must produce BCHW logits for all 32 official classes")
    if min(logits.shape[-2:]) <= 0 or not torch.isfinite(logits).all():
        raise RuntimeError("Prediction model must produce finite logits for all 32 classes")


@torch.inference_mode()
def probabilities(
    model: torch.nn.Module, images: torch.Tensor, output_size: tuple[int, int],
    *, device: torch.device, use_amp: bool, amp_dtype: torch.dtype, hflip_tta: bool,
) -> torch.Tensor:
    """Return CPU CHW probabilities; horizontal flip is undone before averaging."""
    if images.ndim != 4 or images.shape[0] != 1 or images.shape[1] != 3:
        raise ValueError("Image-at-a-time inference requires a single BCHW RGB tensor")
    if len(output_size) != 2 or min(output_size) <= 0:
        raise ValueError("Prediction output_size must be positive [height, width]")
    valid_height, valid_width = images.shape[-2:]
    multiple = getattr(model, "input_multiple", 1)
    if not isinstance(multiple, int) or multiple < 1:
        raise ValueError("Model input_multiple must be a positive integer")
    pad_height = math.ceil(valid_height / multiple) * multiple - valid_height
    pad_width = math.ceil(valid_width / multiple) * multiple - valid_width
    # Replication adds context without stretching the panorama. Remove padding
    # at input resolution before mapping predictions to the native image size.
    padded = F.pad(images, (0, pad_width, 0, pad_height), mode="replicate")

    def forward(inputs: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(inputs)
        _validate_logits(logits, 1)
        if pad_height or pad_width:
            logits = F.interpolate(logits.float(), size=padded.shape[-2:], mode="bilinear", align_corners=False)
            logits = logits[..., :valid_height, :valid_width]
        return F.interpolate(logits.float(), size=output_size, mode="bilinear", align_corners=False).softmax(dim=1)

    result = forward(padded)
    if hflip_tta:
        # Flip the valid image before padding, so the padded margin never moves
        # into the valid field of view when reversing predictions.
        flipped = F.pad(images.flip(-1), (0, pad_width, 0, pad_height), mode="replicate")
        result = (result + forward(flipped).flip(-1)) * 0.5
    return result.squeeze(0).cpu()


@torch.inference_mode()
def panorama_probabilities(
    model: torch.nn.Module, image: torch.Tensor, output_size: tuple[int, int],
    *, device: torch.device, use_amp: bool, amp_dtype: torch.dtype,
    hflip_tta: bool = False, options: InferenceOptions | None = None,
    base_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Average probabilities on CPU, keeping only one image/scale/tile in memory.

    Each scale is sampled directly from the normalized native image; base_size
    is the checkpoint resolution, so large scales retain native image details.
    Whole-image resizing preserves 2:1. Overlapping windows merge probabilities
    with uniform coverage normalization; each window's flip is reversed locally.
    No class-ID masks or logits from different models/scales are averaged.
    """
    options = options or InferenceOptions()
    if image.ndim != 3 or image.shape[0] != 3 or image.device.type != "cpu":
        raise ValueError("Panorama inference expects one normalized CPU CHW RGB tensor")
    if image.shape[-1] != 2 * image.shape[-2]:
        raise ValueError("Inference image must preserve the 2:1 panorama aspect ratio")
    if len(output_size) != 2 or min(output_size) <= 0:
        raise ValueError("Prediction output_size must be positive [height, width]")
    if max(image.shape[-2] * image.shape[-1], output_size[0] * output_size[1]) > options.max_scaled_pixels:
        raise ValueError("Inference image/output exceeds max_scaled_pixels")
    plans = inference_plans(base_size or tuple(image.shape[-2:]), options)
    result = torch.zeros((NUM_CLASSES, *output_size), dtype=torch.float32)
    for plan in plans:
        height, width = plan["size"]
        if (height, width) == tuple(image.shape[-2:]):
            scaled = image
        else:
            scaled = F.interpolate(image.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False).squeeze(0)
        if options.window_size is None:
            probability = probabilities(model, scaled.unsqueeze(0).to(device), output_size,
                                        device=device, use_amp=use_amp, amp_dtype=amp_dtype,
                                        hflip_tta=hflip_tta)
        else:
            accumulated = torch.zeros((NUM_CLASSES, height, width), dtype=torch.float32)
            coverage = torch.zeros((1, height, width), dtype=torch.float32)
            window_height, window_width = options.window_size
            for y in plan["y_starts"]:
                for x in plan["x_starts"]:
                    tile = scaled[:, y:y + window_height, x:x + window_width]
                    tile_height, tile_width = tile.shape[-2:]
                    tile_probability = probabilities(
                        model, tile.unsqueeze(0).to(device), (tile_height, tile_width),
                        device=device, use_amp=use_amp, amp_dtype=amp_dtype, hflip_tta=hflip_tta,
                    )
                    accumulated[:, y:y + tile_height, x:x + tile_width].add_(tile_probability)
                    coverage[:, y:y + tile_height, x:x + tile_width].add_(1)
            if torch.any(coverage == 0):
                raise RuntimeError("Sliding-window inference left uncovered pixels")
            accumulated.div_(coverage)
            probability = F.interpolate(accumulated.unsqueeze(0), size=output_size,
                                        mode="bilinear", align_corners=False).squeeze(0)
        result.add_(probability, alpha=1 / len(plans))
    if not torch.isfinite(result).all():
        raise RuntimeError("Inference probabilities contain nonfinite values")
    return result
