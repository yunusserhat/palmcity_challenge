# PalmCity Challenge

Reproducible semantic segmentation of PalmCity panoramic street images. The current system averages the class probabilities of four EoMT models with DINOv3 ViT-L/16. Three of them use 768 by 1536 pixel inputs and one uses 896 by 1792 pixels. Inference averages scales 1, 1.25 and 1.5 of each model's input size, each with a horizontal flip.

Submission **1020616** obtained **61.54 mIoU and 71.93 mF1** on the hidden test set and ranked first among five participants in the leaderboard snapshot of **30 October 2026**. The same four recipes trained on the training images alone scored **64.53 mIoU and 73.67 mF1** on validation. The submitted models were retrained on the training and validation images together, so they have no independent validation score. See [measured results](docs/results.md) for both stages, the negative results and the limitations.

The first system, from **5 October 2026**, combined three EoMT models at 512 by 1024 pixels. It obtained 57.08 mIoU and 67.96 mF1 on the hidden test set and ranked first among three participants on that date. Its recipes, results and released models remain available.

The repository contains training, evaluation, model adapters, pinned pretrained sources, fixed experiment recipes, a paired image bootstrap and strict submission packaging. It does not contain PalmCity images or model weights. The preparation commands retrieve the official dataset and verify immutable upstream checkpoint files.

## Quick start

Use Linux, Bash, an existing Python 3.12 installation and [uv](https://docs.astral.sh/uv/getting-started/installation/). The locked environment uses PyTorch 2.9.1 with CUDA 12.8. The experiments used one NVIDIA RTX 5090 with 32 GB memory per training job. CPU tests do not require a GPU.

Select a dedicated directory on your own writable storage. Its nearest existing parent must belong to you. A finished training run keeps about 4.8 GiB of checkpoints, 1.2 GiB for `best.pt` and 3.6 GiB for `latest.pt`. Reserve at least **60 GiB** for the three runs of the first system and at least **80 GiB** for the eight runs of the current system, including dependencies, data and intermediate files. The full comparisons need more space.

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

The four final models are available on [Hugging Face](https://huggingface.co/yunusserhat/palmcity-eomt-dinov3-large-v2). After the setup above, you can predict a local panorama or a directory of panoramas directly.

```bash
HF_HUB_OFFLINE=0 bash scripts/run.sh python -m palmcity.hub \
  --download --input /absolute/path/panorama.jpg \
  --output-dir outputs/hub-fast --mode fast --colorize
```

Use `--mode challenge` for the submitted four-model ensemble with scales and horizontal flips, or `--mode single-tta` for one model with the same views. The `fast` mode runs one model at its input size without test time augmentation. Inputs must have the 2:1 panoramic aspect ratio. Outputs include official class-ID PNGs and optional color visualizations. No dataset download or training is needed for this route. The three models of the first system remain available with `--model yunusserhat/palmcity-eomt-dinov3-large --revision addc5a5ef93a26c1d64c1fb23b8c1880ac11c22b`. See the [inference guide](docs/inference.md) for CPU use, offline use and all options.

## Train the current system

Prepare the official data and the eight fixed configs. The `--download` argument and the temporary online flag explicitly enable only the pinned checkpoint download. Training defaults to offline use of those files.

```bash
bash scripts/reproduce.sh data
HF_HUB_OFFLINE=0 bash scripts/reproduce.sh v2-prepare --download
bash scripts/reproduce.sh v2-train
bash scripts/reproduce.sh v2-val
bash scripts/reproduce.sh v2-trainval
bash scripts/reproduce.sh v2-test
```

`v2-train` trains three 768 by 1536 models with seeds 42, 123 and 2026 and one 896 by 1792 model with seed 42 on the training split. `v2-val` scores their ensemble on validation. `v2-trainval` writes a manifest whose training list holds the 497 training and 84 validation images, then retrains the same four recipes for the same 6351 updates without intermediate validation. `v2-test` predicts the test images with the retrained models and writes `$PALMCITY_WORKSPACE/outputs/v2-submission.zip`.

Each recipe uses effective batch size 4, BF16, AdamW, 6351 optimizer updates and the pinned ADE20K EoMT checkpoint. Training flips images horizontally with probability 0.5. With probability 0.5 it also takes a 2:1 crop covering 50 to 100 percent of the panorama area, and with probability 0.8 it changes brightness, contrast, saturation and hue. The stages `prepare`, `train`, `val` and `test` still rebuild the first system.

The ZIP must contain exactly 249 flat PNG files, each 1024 by 512 pixels, in mode `L`, with integer class IDs from 0 through 31. The packager checks these requirements. It does not upload the ZIP.

The default temporary directory is `$PALMCITY_WORKSPACE/tmp`. Its path must be at most 70 bytes because Python multiprocessing creates Unix sockets below it. For a longer workspace path, explicitly set `PALMCITY_TMPDIR` to a short, dedicated directory owned by you on the same filesystem. This opt-in permits temporary IPC files outside the workspace. The scripts never choose an external directory automatically.

For manual training, resume, the comparisons of both stages and faster inference alternatives, follow the [reproduction instructions](docs/reproduction.md).

## Evaluation

The official metric is the macro average over **all 32 classes**, including `Void` with ID 31. A class absent from both ground truth and predictions contributes zero. Labels remain unchanged. No hidden test labels are downloaded or used for model selection.

`python -m palmcity.bootstrap` compares two sets of validation predictions. It draws the 84 validation images with replacement 5000 times, sums their confusion matrices and recomputes the official metric for both sets on the same draws. The interval describes image sampling only. It does not correct for repeated selection on the same images.

The first-stage comparison includes DeepLabV3+ with ResNet50, SegFormer-B2 and B5, UPerNet with ConvNeXt-Large and Swin-Large, Mask2Former with Swin-Large, DINOv3 ViT-B and ViT-L with a linear segmentation decoder, and EoMT with DINOv3 ViT-L. Source initialization differs between practical systems, so the results do not isolate architecture from pretraining.

## License and citation

The repository code is licensed under **GNU General Public License version 3 only**. Dataset and upstream pretrained weights retain their own licenses and access conditions. See [LICENSE](LICENSE), [third-party terms](docs/third-party.md), and [pinned checkpoint provenance](configs/pretrained-sources.json).

The method and both stages are described in [arXiv:2610.06063](https://arxiv.org/abs/2610.06063). Please use [CITATION.cff](CITATION.cff) when citing this work and cite the original PalmCity benchmark.
