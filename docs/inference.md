# Use the released models

The three selected EoMT with DINOv3 ViT-L checkpoints are available from
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
Each seed downloads about 1.26 GB of float32 safe weights. Challenge mode uses all three seeds,
about 3.78 GB in total. Download preflight also reserves room for temporary files.

## Fast single model

This uses the best individual seed, 2026, at the native model resolution with no TTA.
The first command explicitly permits downloading the released model into the existing cache.

```bash
HF_HUB_OFFLINE=0 bash scripts/run.sh python -m palmcity.hub \
  --model yunusserhat/palmcity-eomt-dinov3-large --download \
  --input /absolute/path/panorama.jpg \
  --output-dir outputs/fast --mode fast --device cuda:0 --colorize
```

`outputs/fast/masks` contains grayscale class-ID PNGs with values 0 to 31.
`outputs/fast/colorized` contains optional RGB visualizations in the official PalmCity palette.
`outputs/fast/inference.json` records the chosen model and settings.
Every output path is inside `PALMCITY_WORKSPACE`. Choose a new output folder for a new run.
Use `--device cpu --no-amp` if CUDA is unavailable. The large model will be slower on CPU.

## Single model with TTA

This averages three scales, 0.75, 1.0 and 1.25, with and without horizontal flipping.

```bash
HF_HUB_OFFLINE=0 bash scripts/run.sh python -m palmcity.hub \
  --model yunusserhat/palmcity-eomt-dinov3-large --download \
  --input /absolute/path/panoramas \
  --output-dir outputs/single-tta --mode single-tta --colorize
```

## Challenge ensemble

This uses seeds 42, 123 and 2026 with the same six views per seed as the submitted system.
It averages class probabilities with equal weights, then chooses the most probable class.
One model occupies GPU memory at a time.

```bash
HF_HUB_OFFLINE=0 bash scripts/run.sh python -m palmcity.hub \
  --model yunusserhat/palmcity-eomt-dinov3-large --download \
  --input /absolute/path/panoramas \
  --output-dir outputs/challenge --mode challenge --colorize
```

Once a mode's model files are cached, omit `--download` and the environment override to run
offline. Alternatively pass `--model /absolute/path/to/local/exported-bundle`.
Use `--revision` to select an immutable Hub revision when an exact model version matters.
The released native Transformers files can be loaded with
`EomtDinov3ForUniversalSegmentation.from_pretrained` and a `seed2026` subfolder.
Use the project helper for the measured behavior. It sets the rectangular patch grid,
disables inference attention masks, preserves Void as semantic class 31, and normalizes
query mask scores before probability averaging. The generic Transformers segmentation
pipeline does not supply all of those project-specific steps.

## Measurements

These are original pipeline measurements on 84 PalmCity validation images with an RTX 5090.
They include RGB preprocessing, prediction and PNG writing. The Hub export copies every
float32 model tensor exactly; the helper has been checked against original checkpoints.
Runtime depends on the hardware and input sizes.

| Mode | Validation mIoU | Validation mF1 | Original prediction time |
|---|---:|---:|---:|
| `fast` | 59.67% | 69.72% | 7.85 s |
| `single-tta` | 60.67% | 70.83% | 16.77 s |
| `challenge` | 60.95% | 71.16% | 81.72 s |

The challenge ensemble achieved 57.08% mIoU and 67.96% mF1 on the hidden test leaderboard
on 5 October 2026. The single-model rows are validation results, not hidden test scores.
All results use the published 32-class macro metric, including Void and zero for classes
absent from both truth and prediction. Results on another city or camera may differ.

## Export your own trained checkpoints

The exporter writes native `config.json`, `model.safetensors` and inference metadata,
then compares every saved tensor and dtype to the original model.
It excludes optimizer state, RNG state and private training paths.

```bash
bash scripts/run.sh python -m palmcity.export_hub \
  --checkpoint /absolute/path/seed42/best.pt \
               /absolute/path/seed123/best.pt \
               /absolute/path/seed2026/best.pt \
  --output-dir outputs/hub-export
```

License and attribution files must accompany any further distribution of DINOv3 derivatives.
Exporting weights does not grant additional rights to upstream data or models.
