# Use the released models

The four final EoMT with DINOv3 ViT-L models of the current system are available from
[yunusserhat/palmcity-eomt-dinov3-large-v2](https://huggingface.co/yunusserhat/palmcity-eomt-dinov3-large-v2).
The three models of the first system remain at
[yunusserhat/palmcity-eomt-dinov3-large](https://huggingface.co/yunusserhat/palmcity-eomt-dinov3-large).
You can predict your own local panoramic images without downloading PalmCity or running training.
Read the model repository's DINOv3 license before using the weights. The GitHub code remains GPLv3.

## Setup

Clone this repository and follow the environment setup in the [README](../README.md).
Choose an absolute `PALMCITY_WORKSPACE` for outputs and keep it set in the commands below.
The existing Worf Hugging Face cache is reused automatically on Worf.
On other systems the setup places the cache in the selected workspace unless you explicitly
configure another existing cache.

Inputs are RGB images with a 2:1 aspect ratio, such as 1024 by 512 panoramic street views.
The CLI accepts one image or a directory of images. It preserves each image's output resolution.
Image names in a directory must have unique stems. No manifest or ground truth is required.
Each model downloads about 1.26 GB of float32 safe weights. Challenge mode of the current release
uses four models, about 5 GB in total. Download preflight also reserves room for temporary files.

## Release members

| Member | Training input | Seed | Training images |
|---|---|---:|---:|
| `h1-seed42` | 768 by 1536 | 42 | 497 training and 84 validation |
| `h1-seed123` | 768 by 1536 | 123 | 497 training and 84 validation |
| `h1-seed2026` | 768 by 1536 | 2026 | 497 training and 84 validation |
| `h15-seed42` | 896 by 1792 | 42 | 497 training and 84 validation |

The `fast` and `single-tta` modes use `h1-seed2026` unless you pass `--member`.
Its training-only counterpart had the highest validation score of the three 768 by 1536 seeds.
Each member keeps its own input size. Scales are relative to that size.

## Fast single model

This runs one member at its input size with no test time augmentation.
The first command explicitly permits downloading the released model into the existing cache.

```bash
HF_HUB_OFFLINE=0 bash scripts/run.sh python -m palmcity.hub \
  --download --input /absolute/path/panorama.jpg \
  --output-dir outputs/fast --mode fast --device cuda:0 --colorize
```

`outputs/fast/masks` contains grayscale class-ID PNGs with values 0 to 31.
`outputs/fast/colorized` contains optional RGB visualizations in the official PalmCity palette.
`outputs/fast/inference.json` records the chosen members and settings.
Every output path is inside `PALMCITY_WORKSPACE`. Choose a new output folder for a new run.
Use `--device cpu --no-amp` if CUDA is unavailable. The large model will be slower on CPU.

## Single model with test time augmentation

This averages scales 1, 1.25 and 1.5 of the member's input size, with and without horizontal flipping.

```bash
HF_HUB_OFFLINE=0 bash scripts/run.sh python -m palmcity.hub \
  --download --input /absolute/path/panoramas \
  --output-dir outputs/single-tta --mode single-tta --colorize
```

## Challenge ensemble

This uses all four members with the same six views per member as the submitted system.
It averages class probabilities with equal weights, then chooses the most probable class.
One model occupies GPU memory at a time. The release metadata raises the per-scale probability
map bound to 4,000,000 pixels so that the largest view of the 896 by 1792 member fits.

```bash
HF_HUB_OFFLINE=0 bash scripts/run.sh python -m palmcity.hub \
  --download --input /absolute/path/panoramas \
  --output-dir outputs/challenge --mode challenge --colorize
```

## First system

The first release uses seeds 42, 123 and 2026 at 512 by 1024 pixels with scales 0.75, 1 and 1.25.
Name its repository and pin its immutable revision to repeat the 5 October 2026 results.

```bash
HF_HUB_OFFLINE=0 bash scripts/run.sh python -m palmcity.hub \
  --model yunusserhat/palmcity-eomt-dinov3-large \
  --revision addc5a5ef93a26c1d64c1fb23b8c1880ac11c22b --download \
  --input /absolute/path/panoramas \
  --output-dir outputs/challenge-first --mode challenge --colorize
```

Its single-model modes use seed 2026 unless you pass `--seed` or `--member` with a folder such as `seed42`.

## Offline use and native files

Once a mode's model files are cached, omit `--download` and the environment override to run
offline. Alternatively pass `--model /absolute/path/to/local/exported-bundle`.
Use `--revision` to select an immutable Hub revision when an exact model version matters.
The released native Transformers files can be loaded with
`EomtDinov3ForUniversalSegmentation.from_pretrained` and a member subfolder such as `h1-seed2026`.
Use the project helper for the measured behavior. It sets the rectangular patch grid,
disables inference attention masks, preserves Void as semantic class 31, and normalizes
query mask scores before probability averaging. The generic Transformers segmentation
pipeline does not supply all of those project-specific steps.

## Measurements

The submitted four-model ensemble predicted the 249 test images in 1070.92 seconds on one RTX 5090,
or 1088.71 seconds including checkpoint loading and provenance checks. Peak allocated GPU memory
was 5.63 GiB. It obtained 61.54% mIoU and 71.93% mF1 on the hidden test leaderboard on 30 October 2026.
The released models were trained on the validation images, so validation scores of these files are
not meaningful. Their training-only counterparts scored 64.53% mIoU and 73.67% mF1 on validation.

The first release was measured on 84 PalmCity validation images with an RTX 5090.
The times include RGB preprocessing, prediction and PNG writing.

| Mode, first release | Validation mIoU | Validation mF1 | Prediction time |
|---|---:|---:|---:|
| `fast` | 59.67% | 69.72% | 7.85 s |
| `single-tta` | 60.67% | 70.83% | 16.77 s |
| `challenge` | 60.95% | 71.16% | 81.72 s |

Its challenge ensemble achieved 57.08% mIoU and 67.96% mF1 on the hidden test leaderboard
on 5 October 2026. All results use the published 32-class macro metric, including Void and zero
for classes absent from both truth and prediction. Results on another city or camera may differ.

## Export your own trained checkpoints

The exporter writes native `config.json`, `model.safetensors` and inference metadata for each member,
then compares every saved tensor and dtype to the original model.
It excludes optimizer state, RNG state and private training paths.
The root `palmcity_inference.json` lists the members, their input sizes and the challenge settings.

```bash
bash scripts/run.sh python -m palmcity.export_hub \
  --checkpoint "$PALMCITY_WORKSPACE/runs/final-h1-trainval-seed42/best.pt" \
               "$PALMCITY_WORKSPACE/runs/final-h1-trainval-seed123/best.pt" \
               "$PALMCITY_WORKSPACE/runs/final-h1-trainval-seed2026/best.pt" \
               "$PALMCITY_WORKSPACE/runs/final-h15-trainval-seed42/best.pt" \
  --name h1-seed42 h1-seed123 h1-seed2026 h15-seed42 \
  --scales 1 1.25 1.5 --max-scaled-pixels 4000000 --single-model h1-seed2026 \
  --output-dir outputs/hub-export
```

Without `--name`, members are called `seed<N>` and the first-release settings are the defaults.
License and attribution files must accompany any further distribution of DINOv3 derivatives.
Exporting weights does not grant additional rights to upstream data or models.
