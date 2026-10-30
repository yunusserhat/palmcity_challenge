# Reproducing the experiments

All commands run from the repository root. Set `PALMCITY_WORKSPACE` to an absolute dedicated directory owned by your user, and keep it set for the entire workflow. Runtime outputs remain below that directory. The scripts do not change shell startup files, install into system Python, or invoke `pip --user`.

The sections up to the comparison of nine models rebuild the system of 5 October 2026. The second stage section rebuilds the system of 30 October 2026. Both stages use the same locked environment.

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

The storage preflight verifies ownership, actual write access, filesystem identity, free space and output confinement. Dependency installation reserves 20 GiB of free space. Dataset preparation reserves 5 GiB. Three selected training runs should start with at least 60 GiB available before environment setup. The eight runs of the second stage need at least 80 GiB, because each finished run keeps about 4.8 GiB of checkpoints. Keeping all candidates and optimizer states can require more than 150 GiB. Peak space depends on which intermediate runs you retain.

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

## Second stage, 28 to 30 October 2026

The second stage kept the EoMT architecture, the pinned ADE20K checkpoint and the 6351-update budget of the first system and changed the training recipe. `configs/recipes` contains the 16 measured EoMT recipes of this stage. Each one equals the recorded run configuration after removing its local checkpoint path.

| Recipes | Change from the EoMT recipe of the first system |
|---|---|
| `campaign-e1-augment-seed{42,123,2026}` | A 2:1 crop over 50 to 100 percent of the panorama area with probability 0.5, and brightness, contrast, saturation and hue changes of strength 0.2, 0.2, 0.2 and 0.05 with probability 0.8 |
| `campaign-e2-sampling-seed42` | Training images drawn by rare class presence in the training masks, with power 0.5, maximum weight 5 and at least 16 pixels. Road, Building, Sky and Void do not count toward rarity |
| `campaign-e3-augment-sampling-seed42` | Both changes |
| `campaign-p1-points4x-seed42` | 50176 instead of 12544 matching points. The measured run stopped after 5000 of 6351 updates |
| `campaign-h1-augment-768x1536-seed{42,123,2026,7,99}` | The E1 changes at 768 by 1536 pixels |
| `campaign-h15-augment-896x1792-seed42` | The E1 changes at 896 by 1792 pixels |
| `final-h1-trainval-seed{42,123,2026}`, `final-h15-trainval-seed42` | The selected recipes for the train plus val manifest, without intermediate validation |

Two chains of runs selected the system on validation with rules written into their scripts before the results existed. The first chain trained the H1 seeds 123 and 2026 and compared three H1 models with the three E1 models plus the three H1 models, using scales 0.75, 1 and 1.25. The rule kept the higher validation mIoU, the three H1 models with 63.75. The second chain first kept the scale set with the highest validation mIoU among four sets for the three H1 models, which was 1, 1.25 and 1.5 with 64.01. It then trained the H1 seeds 7 and 99 and the H15 model and compared three or five H1 models, each with and without the H15 model. The rule again kept the highest validation mIoU, three H1 models with the H15 model at 64.53. The selected recipes were then retrained on the training and validation images for the same number of updates.

The `v2-*` stages of `scripts/reproduce.sh` repeat the selected path, as shown in the [README](../README.md). The train plus val manifest and a paired comparison can also be produced directly.

```bash
bash scripts/run.sh python -m palmcity.data trainval \
  --manifest "$PALMCITY_WORKSPACE/manifests/dataset.json" \
  --output manifests/trainval.json --strict-counts

bash scripts/run.sh python -m palmcity.bootstrap \
  --manifest "$PALMCITY_WORKSPACE/manifests/dataset.json" \
  --baseline-predictions "$PALMCITY_WORKSPACE/outputs/val-ensemble-tta" \
  --baseline-report "$PALMCITY_WORKSPACE/reports/val-ensemble-tta-score.json" \
  --candidate-predictions "$PALMCITY_WORKSPACE/outputs/v2-val-ensemble" \
  --report reports/second-vs-first-bootstrap.json
```

The manifest keeps the training list first and appends the validation images, so the training list holds 581 images. Its validation and test lists are unchanged. A validation score computed after training on it only shows that training ran. The bootstrap draws the validation images with replacement 5000 times with seed 20261028 and reports the candidate minus baseline difference, its 2.5 and 97.5 percentiles and the share of draws with a positive difference.

The H15 recipe needs most of a 32 GB card. Its measured run reached 26.14 GiB of peak allocated memory, and a short benchmark reserved 28.34 GiB. A 1024 by 2048 benchmark ran out of memory at 24.20 GiB allocated, and the Transformers EoMT model in the locked environment does not support gradient checkpointing. On one RTX 5090 an H1 run took about 97 minutes and the H15 run about 145 minutes. The final test prediction with four models used 5.63 GiB of peak allocated memory and needs `--max-scaled-pixels 4000000` for the 1.5 scale of the H15 model.

Four further systems and two post processing methods were measured in this stage and are reported in the [results](results.md), but they are not part of this release. They are SegFormer-B5 from a Cityscapes checkpoint, a longer Mask2Former run, an Ultralytics YOLO26x semantic model, EoMT initialized from a LightlyTrain Cityscapes checkpoint, SAM voting as in Semantic Segment Anything, and a YOLOv8x-seg mask overlay. Several of them need separate environments or AGPL licensed packages, and none of them improved the validation score.

## Download-free verification

```bash
bash scripts/verify.sh
PALMCITY_VERIFY_GPU=1 bash scripts/verify.sh
```

The default check runs Ruff and the CPU test suite using synthetic local images, masks, checkpoints and random model weights. It does not download models or real data. The optional GPU check adds random-weight BF16 forward passes. Tests cover native and query segmentation adapters, official metric edge cases, probability averaging, TTA alignment, resumable fixed budgets, strict ZIP requirements, pinned-file hashes and portable storage confinement. They also cover the training augmentation and rare class sampling with checkpoint resume, the paired bootstrap, the train plus val manifest and releases whose members use different input sizes.
