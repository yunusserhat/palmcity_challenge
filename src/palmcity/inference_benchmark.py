"""Benchmark inference mechanics using random tiny weights and synthetic pixels only."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from .benchmark import idle_gpu
from .inference import InferenceOptions, inference_plans, panorama_probabilities
from .models import TinySegmentationModel, amp_settings, select_device
from .storage import require_workspace, safe_output_path
from .training_state import code_identity, dependency_versions


def synthetic_benchmark(
    report_path: str | Path, *, device_name: str = "cpu", image_size: tuple[int, int] = (512, 1024),
    repeats: int = 3, seed: int = 17,
) -> Path:
    """Measure TTA/tiling mechanics; these timings do not describe candidate models."""
    workspace = require_workspace()
    report_path = safe_output_path(report_path, workspace)
    if report_path.exists():
        raise FileExistsError(f"Benchmark report already exists: {report_path}")
    if repeats < 1 or repeats > 20:
        raise ValueError("Synthetic benchmark repeats must be between 1 and 20")
    inference_plans(image_size, InferenceOptions())
    device_spec = torch.device(device_name)
    gpu_preflight = None
    if device_spec.type == "cuda":
        if device_spec.index is None:
            raise ValueError("Synthetic GPU inference requires explicit physical cuda:N")
        # Check physical occupancy before select_device initializes a CUDA
        # context, so our own process cannot obscure the shared-GPU preflight.
        gpu_preflight = idle_gpu(device_spec.index)
        print(json.dumps(gpu_preflight), flush=True)
    device = select_device(device_name)
    torch.manual_seed(seed)
    model = TinySegmentationModel().eval().to(device)
    image = torch.randn((3, *image_size), generator=torch.Generator().manual_seed(seed))
    use_amp, dtype = amp_settings(device, True)
    window_height = max(1, image_size[0] // 2)
    scenarios = [
        ("single_full", InferenceOptions(), False),
        ("full_flip", InferenceOptions(), True),
        ("multiscale_flip", InferenceOptions(scales=(0.75, 1.0, 1.25)), True),
        ("multiscale_window_flip", InferenceOptions(scales=(0.75, 1.0, 1.25),
                                                  window_size=(window_height, 2 * window_height)), True),
    ]
    for _, options, _ in scenarios:
        inference_plans(image_size, options)
    rows = []
    for name, options, hflip in scenarios:
        def forward():
            result = panorama_probabilities(model, image, image_size, device=device, use_amp=use_amp,
                                            amp_dtype=dtype, options=options, hflip_tta=hflip)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            return result

        # Warm up every scenario at its actual shape before comparing latency.
        forward()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        durations = []
        for _ in range(repeats):
            started = time.perf_counter()
            result = forward()
            durations.append(time.perf_counter() - started)
            if result.shape != (32, *image_size) or not torch.isfinite(result).all():
                raise RuntimeError("Synthetic benchmark produced invalid 32-class probabilities")
            torch.testing.assert_close(result.sum(0), torch.ones(image_size), atol=1e-5, rtol=1e-5)
        row = {"name": name, "options": asdict(options), "hflip": hflip,
               "seconds": durations, "mean_seconds": statistics.mean(durations),
               "median_seconds": statistics.median(durations),
               "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None,
               "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else None,
               "all_probabilities_finite_normalized": True}
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "scope": "synthetic random TinySegmentationModel inference only; not candidate architecture throughput or a measured PalmCity score",
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_preflight": gpu_preflight, "seed": seed, "shape": list(image_size),
        "amp": use_amp, "dtype": str(dtype) if use_amp else None,
        "cpu_threads": torch.get_num_threads(), "repeats": repeats,
        "warmup": "one complete forward per scenario at actual shapes; excluded from timings",
        "code": code_identity(), "dependencies": dependency_versions(), "scenarios": rows,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="A new report path inside PALMCITY_WORKSPACE")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", nargs=2, type=int, default=[512, 1024], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    report = synthetic_benchmark(args.report, device_name=args.device, image_size=tuple(args.image_size),
                                 repeats=args.repeats, seed=args.seed)
    print(report)


if __name__ == "__main__":
    main()
