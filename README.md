# PalmCity Challenge

Reproducible semantic segmentation of PalmCity panoramic street images. The selected system combines three EoMT models with DINOv3 ViT-L/16, probability averaging, three image scales and horizontal flip augmentation at inference.

Submission **961441** obtained **57.08 mIoU and 67.96 mF1** on the hidden test set. It ranked first among three participants in the leaderboard snapshot dated **5 October 2026**. The public validation result was **60.95 mIoU and 71.16 mF1**. See [measured results](docs/results.md) for the experiments, timings and limitations.

The repository contains training, evaluation, model adapters, pinned pretrained sources, fixed experiment recipes and strict submission packaging. It does not contain PalmCity images or model weights. The preparation commands retrieve the official dataset and verify immutable upstream checkpoint files.

## Quick start

Use Linux, Bash, an existing Python 3.12 installation and [uv](https://docs.astral.sh/uv/getting-started/installation/). The locked environment uses PyTorch 2.9.1 with CUDA 12.8. The experiments used one NVIDIA RTX 5090 with 32 GB memory per training job. CPU tests do not require a GPU.

Select a dedicated directory on your own writable storage. Its nearest existing parent must belong to you. Reserve at least **60 GiB** for the selected three-seed training workflow, including dependencies, optimizer states, weights and intermediate files. The full comparison needs more space.

```bash
git clone https://github.com/yunusserhat/palmcity_challenge.git
cd palmcity_challenge
export PALMCITY_WORKSPACE="/absolute/path/to/your/palmcity-workspace"
export PALMCITY_PYTHON="python3.12"
# For deeply nested workspaces, also select a short directory on the same filesystem.
# export PALMCITY_TMPDIR="/absolute/short/user-owned/palmcity-tmp"
bash scripts/setup.sh
bash scripts/verify.sh
```

## Use the trained models

The three final EoMT models are available on [Hugging Face](https://huggingface.co/yunusserhat/palmcity-eomt-dinov3-large). After the setup above, you can predict a local panorama or a directory of panoramas directly.

```bash
HF_HUB_OFFLINE=0 bash scripts/run.sh python -m palmcity.hub \
  --download --input /absolute/path/panorama.jpg \
  --output-dir outputs/hub-fast --mode fast --colorize
```

Use `--mode challenge` for the submitted three-model ensemble with scales and horizontal flip, or `--mode single-tta` for the faster transformed single model. Inputs must have the 2:1 panoramic aspect ratio. Outputs include official class-ID PNGs and optional color visualizations. No dataset download or training is needed for this inference route. See the [inference guide](docs/inference.md) for CPU use, offline use and all options.

## Train the selected system

Prepare the official data and the three selected configs. The `--download` argument and the temporary online flag explicitly enable only the pinned checkpoint download. Training defaults to offline use of those files.

```bash
bash scripts/reproduce.sh data
HF_HUB_OFFLINE=0 bash scripts/reproduce.sh prepare --download
bash scripts/reproduce.sh train
bash scripts/reproduce.sh val
bash scripts/reproduce.sh test
```

The final file is `$PALMCITY_WORKSPACE/outputs/submission.zip`. It must contain exactly 249 flat PNG files, each 1024 by 512 pixels, in mode `L`, with integer class IDs from 0 through 31. The packager checks these requirements. It does not upload the ZIP.

The default temporary directory is `$PALMCITY_WORKSPACE/tmp`. Its path must be at most 70 bytes because Python multiprocessing creates Unix sockets below it. For a longer workspace path, explicitly set `PALMCITY_TMPDIR` to a short, dedicated directory owned by you on the same filesystem. This opt-in permits temporary IPC files outside the workspace; the scripts never choose an external directory automatically.

The selected recipe uses seeds 42, 123 and 2026, full 512 by 1024 panoramas, effective batch size 4, BF16, AdamW, 6351 optimizer updates and ten validation checks. Models are trained separately from the same pretrained source. The inference ensemble gives each model equal weight, averages probabilities over scales 0.75, 1 and 1.25 and also uses their horizontally flipped versions.

For manual training, resume, the nine-model comparison and faster inference alternatives, follow [reproduction instructions](docs/reproduction.md).

## Evaluation

The official metric is the macro average over **all 32 classes**, including `Void` with ID 31. A class absent from both ground truth and predictions contributes zero. Labels remain unchanged. No hidden test labels are downloaded or used for model selection.

The comparison includes DeepLabV3+ with ResNet50, SegFormer-B2 and B5, UPerNet with ConvNeXt-Large and Swin-Large, Mask2Former with Swin-Large, DINOv3 ViT-B and ViT-L with a linear segmentation decoder, and EoMT with DINOv3 ViT-L. Source initialization differs between practical systems, so the results do not isolate architecture from pretraining.

## License and citation

The repository code is licensed under **GNU General Public License version 3 only**. Dataset and upstream pretrained weights retain their own licenses and access conditions. See [LICENSE](LICENSE), [third-party terms](docs/third-party.md), and [pinned checkpoint provenance](configs/pretrained-sources.json).

Please use [CITATION.cff](CITATION.cff) when citing this repository and cite the original PalmCity benchmark.
