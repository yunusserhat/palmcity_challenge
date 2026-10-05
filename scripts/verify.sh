#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$PALMCITY_CODE_ROOT"
bash -n scripts/env.sh scripts/setup.sh scripts/run.sh scripts/verify.sh
"$UV_PROJECT_ENVIRONMENT/bin/ruff" check src tests
"$UV_PROJECT_ENVIRONMENT/bin/python" -m pytest -q
# Optional GPU smoke checks use synthetic files and random model weights.
if [[ "${PALMCITY_VERIFY_GPU:-0}" != 1 ]]; then
    exit 0
fi
"$UV_PROJECT_ENVIRONMENT/bin/python" - <<'PY'
import torch
import segmentation_models_pytorch as smp

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable: CPU tests passed; requested GPU verification incomplete.")
for architecture, encoder in [("DeepLabV3Plus", "resnet50"), ("Segformer", "mit_b2")]:
    model = getattr(smp, architecture)(
        encoder_name=encoder, encoder_weights=None, in_channels=3, classes=32,
    ).eval().to("cuda:0")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(torch.zeros(1, 3, 64, 128, device="cuda:0"))
    assert output.shape == (1, 32, 64, 128)
    assert torch.isfinite(output).all()
    torch.cuda.synchronize()
    print(f"{architecture}/{encoder}: CUDA bfloat16 forward passed; no weights downloaded.")
    del model, output
PY
