# Reproducing the experiments

All commands run from the repository root. Set `PALMCITY_WORKSPACE` to an absolute dedicated directory owned by your user, and keep it set for the entire workflow. Runtime outputs remain below that directory. The scripts do not change shell startup files, install into system Python, or invoke `pip --user`.

## Environment and storage

```bash
export PALMCITY_WORKSPACE="/absolute/path/to/your/palmcity-workspace"
export PALMCITY_PYTHON="python3.12"
bash scripts/setup.sh
bash scripts/run.sh python -m palmcity.storage --min-free-gib 50
bash scripts/verify.sh
```

Install Python 3.12 and uv separately before setup. `PALMCITY_PYTHON` can be an absolute interpreter path. `uv sync --locked` uses the supplied `uv.lock` without dependency re-resolution. Python downloads are disabled. The project name in the lock preserves the original environment metadata.

The default environment is `$PALMCITY_WORKSPACE/.venv`. `PALMCITY_ENVIRONMENT` selects an existing environment explicitly, which is useful for local verification. Setup synchronizes the selected environment, so it should belong to this project. `PALMCITY_DEVICE`, such as `cuda:0`, selects the GPU for `scripts/reproduce.sh`. Runtime scripts require a prepared environment. The expected training device supports BF16; the original runs used CUDA 12.8 and one 32 GB RTX 5090 per job.

The storage preflight verifies ownership, actual write access, filesystem identity, free space and output confinement. Dependency installation reserves 20 GiB of free space. Dataset preparation reserves 5 GiB. Three selected training runs should start with at least 60 GiB available before environment setup. Keeping all candidates and optimizer states can require more than 150 GiB. Peak space depends on which intermediate runs you retain.

Python multiprocessing creates Unix-domain sockets below its temporary directory. The default `$PALMCITY_WORKSPACE/tmp` must be at most 70 bytes long, including the full absolute path. A longer path is rejected before multiworker training begins. For a deeply nested workspace, explicitly select a short temporary directory before setup and verification.

```bash
export PALMCITY_TMPDIR="/absolute/short/user-owned/palmcity-tmp"
```

This is an explicit exception to workspace-only temporary writes. The directory must have a user-owned writable parent, remain on the same filesystem as the workspace, and pass an actual write probe. On Worf it must remain on scratch. The scripts never choose another directory automatically, fall back to the system temporary directory, change shared permissions or move existing files. Training artifacts, data, configs, reports and caches retain their configured locations.

On Worf, the scripts automatically source the existing `~/.config/worf/huggingface-cache.sh`, reuse its Hub cache and require the workspace on the mounted scratch filesystem. They do not create a second Hub cache. Elsewhere, the default Hub cache is `$PALMCITY_WORKSPACE/cache/huggingface/hub`; an existing absolute `HF_HUB_CACHE` can be supplied explicitly. Authentication stays outside the repository. Gated DINOv3 sources require accepting the upstream conditions and authenticating with Hugging Face using its normal login workflow.

## Official data and unchanged labels

```bash
bash scripts/reproduce.sh data
```

This retrieves only official public image folders and training/validation masks, then builds and audits the manifest. The official split has 497 training images, 84 validation images and 249 test images. Each image is 1024 by 512 pixels. Masks contain direct class IDs 0 through 31. The data downloader never requests hidden test annotations. It retains a per-file SHA-256 inventory in the workspace.

For an existing official dataset, build its manifest explicitly instead of downloading again.

```bash
bash scripts/run.sh python -m palmcity.data build \
  --root "/absolute/path/to/PalmCity" \
  --output manifests/dataset.json \
  --report reports/dataset-audit.json --strict-counts
```

The expected layout is `images/{train,val,test}` and `annotations/gt/{train,val}`. Nonstandard mask directories can be supplied with `--mask-dir train=/absolute/path` and `--mask-dir val=/absolute/path`. The manifest records absolute paths for the current machine. Consequently, manifest and path-dependent identity hashes will differ between workspaces even when file content is identical.

## Immutable pretrained initialization

Every measured candidate has a source repository, immutable revision, expected filenames, file lengths and SHA-256 values in `configs/pretrained-sources.json`. Portable recipes contain no machine-specific checkpoint paths. The preparer resolves the source in the selected cache, verifies every file and writes a new local config with a provenance sidecar.

```bash
HF_HUB_OFFLINE=0 bash scripts/reproduce.sh prepare --download
```

If the pinned snapshots are already cached, omit the download opt-in.

```bash
bash scripts/reproduce.sh prepare
```

Preparation refuses existing output configs. Run it once per workspace, or choose different output filenames for independent experiments. A manually prepared candidate can use the same CLI.

```bash
HF_HUB_OFFLINE=0 bash scripts/run.sh python -m palmcity.pretrained \
  --recipe configs/candidates/dinov3_vit_large_linear.json \
  --output configs/dinov3-large.json --download
```

SMP encoders load the verified local safetensors file directly. Transformers models use the verified snapshot directory. The preparer does not fetch a moving `main` revision. The 32-class PalmCity classifier is initialized for transfer while compatible pretrained features remain intact. Prediction reconstructs saved architecture without fetching pretrained weights.

The GPL license covers the repository code. It does not relicense pretrained weights, the benchmark dataset or their derivatives. SegFormer, Mask2Former and DINOv3 sources have additional upstream conditions. Read their linked terms before using them.

## Exact selected training budget

```bash
bash scripts/reproduce.sh train
```

The command trains three independent EoMT models from the pinned ADE20K checkpoint. Their seeds are 42, 123 and 2026. Each fixed config uses the original 6351-update budget, 51 epochs as an upper bound, microbatch size 1, four accumulated microbatches, learning rate 0.00006, minimum learning rate 0.000001, polynomial power 0.9, 317 warmup updates, weight decay 0.01 and gradient norm limit 1.0. AdamW excludes biases and normalization parameters from weight decay. Uniform backbone/head learning rate is the PalmCity adaptation used in the reported runs.

Horizontal flip probability is 0.5 during training. No panorama roll is used. Inputs use ImageNet channel normalization. The native query loss includes mask, Dice and class supervision with Hungarian assignment. The query no-object token is separate from PalmCity `Void`. The four-block attention-mask schedule is retained in the config and ends with unmasked inference. Validation runs after epochs 6, 11, 16, 21, 26, 31, 36, 41, 46 and 51. The final epoch can stop at the fixed update cap.

Each run writes `best.pt`, `latest.pt`, configuration, metrics and provenance below `$PALMCITY_WORKSPACE/runs`. The best checkpoint is selected using native validation mIoU at the ten declared checks. `latest.pt` includes optimizer, scheduler, scaler, RNG and data-loader state. Epoch-boundary resume preserves the original update schedule and replays an interrupted, uncommitted epoch.

To pause and resume a manual run, use the same config and run name.

```bash
bash scripts/run.sh python -m palmcity.train \
  --config "$PALMCITY_WORKSPACE/configs/eomt-seed42.json" \
  --manifest "$PALMCITY_WORKSPACE/manifests/dataset.json" \
  --run-name confirmation-eomt_dinov3_vit_large-seed42 \
  --device cuda:0 --allow-pretrained-downloads --stop-after-epochs 2

bash scripts/run.sh python -m palmcity.train \
  --config "$PALMCITY_WORKSPACE/configs/eomt-seed42.json" \
  --manifest "$PALMCITY_WORKSPACE/manifests/dataset.json" \
  --run-name confirmation-eomt_dinov3_vit_large-seed42 \
  --device cuda:0 --allow-pretrained-downloads \
  --resume "$PALMCITY_WORKSPACE/runs/confirmation-eomt_dinov3_vit_large-seed42/latest.pt"
```

Resume rejects changes to source, dependencies, data, configuration or schedule. The portable release has its own source identity. It does not claim to be byte-identical to the frozen original implementation. Original measured scores came from that frozen implementation; this release preserves its model/training/inference mathematics while changing storage portability and checkpoint preparation. BF16 kernels and nondeterministic GPU operations can cause variation between retraining runs. Reproduction means executing the documented method and budget, not promising the same ZIP hash.

## Validation, inference and strict packaging

```bash
bash scripts/reproduce.sh val
bash scripts/reproduce.sh test
```

Each trained model receives equal ensemble weight. For every image, predictions from scales 0.75, 1 and 1.25 and both horizontal orientations are aligned to native resolution and averaged as probabilities. The ensemble average is converted to class IDs by argmax. CPU probability accumulation keeps only one model on the GPU at a time. Output sidecars retain checkpoint, data and prediction hashes, timings and peak allocated/reserved GPU memory.

Validation uses the published 32-class macro mIoU and mF1 convention. `Void` contributes like any other class. A class absent from both ground truth and predictions contributes zero. Model selection, ensemble weights and TTA settings use only the public validation split.

The strict packager requires exactly 249 matching flat PNG filenames, mode `L`, size 1024 by 512 and IDs 0 through 31. `--allow-partial` is reserved for synthetic tests and is never used for official submission.

A faster alternative uses only the seed-2026 model with the same TTA. In the original measurements it reached 60.67 validation mIoU, compared with 60.95 for the selected three-model TTA ensemble.

```bash
bash scripts/run.sh python -m palmcity.predict \
  --checkpoint "$PALMCITY_WORKSPACE/runs/confirmation-eomt_dinov3_vit_large-seed2026/best.pt" \
  --manifest "$PALMCITY_WORKSPACE/manifests/dataset.json" \
  --output-dir outputs/val-single-tta --split val --device cuda:0 \
  --scales 0.75 1 1.25 --hflip-tta
```

For native single-model inference omit `--scales` and `--hflip-tta`. For native three-model averaging pass three `--checkpoint` paths and `--weights 1 1 1` without TTA. Overlapping window inference is available through `--window-size 256 512 --overlap 0.5`, but it was not the selected submission method.

## Nine-model comparison

`configs/candidates` contains the nine eligible pretrained recipes. `configs/recipes` contains their original fixed pilot budgets and six longer confirmations for EoMT-L and DINOv3-L. The fixed pilot recipes were calibrated to approximately 600 seconds of training plus validation on the original RTX 5090. Confirmation recipes used approximately 2400 seconds. Startup and checkpoint I/O were excluded from those targets. The fixed update counts reproduce that hardware's chosen schedules, but they do not guarantee equal seconds on a different GPU.

To repeat the original fixed pilot schedule, prepare any `configs/recipes/pilot-*.json` with `palmcity.pretrained`, then train it with `palmcity.train`. To make a fair compute comparison on new hardware, recalibrate all nine candidate configs using real image decoding, native loss, backward and optimizer updates. The following loop creates fresh prepared configs and calibration reports.

```bash
for recipe in configs/candidates/*.json; do
  candidate="$(basename "$recipe" .json)"
  HF_HUB_OFFLINE=0 bash scripts/run.sh python -m palmcity.pretrained \
    --recipe "$recipe" --output "configs/candidate-$candidate.json" --download
  bash scripts/run.sh python -m palmcity.experiments calibrate \
    --config "$PALMCITY_WORKSPACE/configs/candidate-$candidate.json" \
    --manifest "$PALMCITY_WORKSPACE/manifests/dataset.json" \
    --report "reports/calibration/$candidate.json" \
    --device cuda:0 --allow-pretrained-downloads
 done

bash scripts/run.sh python -m palmcity.experiments plan \
  --calibrations "$PALMCITY_WORKSPACE"/reports/calibration/*.json \
  --output-dir reports/pilot-plan --phase pilot --seconds 600 --evaluations 10

bash scripts/run.sh python -m palmcity.experiments run \
  --plan "$PALMCITY_WORKSPACE/reports/pilot-plan/plan.json" --device cuda:0

bash scripts/run.sh python -m palmcity.experiments aggregate \
  --plan "$PALMCITY_WORKSPACE/reports/pilot-plan/plan.json" \
  --audit "$PALMCITY_WORKSPACE/reports/dataset-audit.json" \
  --report reports/pilot-results.json
```

The planner refuses budgets too small for the reserved validation checks. Raise the declared budget for every candidate together when necessary. Keep resolution, effective batch size 4, BF16, input data and validation opportunities equal. Select two candidates from these public validation results, then create a confirmation plan with their two calibration reports, `--phase confirmation --seconds 2400 --seeds 42 123 2026`. Run and aggregate it through the same commands.

Different pretrained sources and source-informed optimizer choices make this a comparison of complete practical systems. It is not a controlled claim that architecture alone caused the ranking. The ConvNeXt scalar backbone learning-rate multiplier approximates the upstream layer-wise policy. Four superseded optimizer recipes were diagnostic runs and were excluded from the eligible model ranking.

Their original fixed budgets are available separately in `configs/diagnostics`. Each carries `publication_selection_status` set to `excluded_from_primary_selection`. Prepare and train them manually only to recreate the four historical diagnostics. Together with the nine eligible pilots and six confirmations, these recipes cover all 19 reported training runs. Do not select the maximum of a superseded recipe and its corrected counterpart as a primary model comparison.

## Download-free verification

```bash
bash scripts/verify.sh
PALMCITY_VERIFY_GPU=1 bash scripts/verify.sh
```

The default check runs Ruff and the CPU test suite using synthetic local images, masks, checkpoints and random model weights. It does not download models or real data. The optional GPU check adds random-weight BF16 forward passes. Tests cover native and query segmentation adapters, official metric edge cases, probability averaging, TTA alignment, resumable fixed budgets, strict ZIP requirements, pinned-file hashes and portable storage confinement.
