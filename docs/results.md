# Recorded challenge results

## Second stage, 28 to 30 October 2026

On 20 October 2026 another entry reached 60.55 mIoU on the public leaderboard. The second stage then tested recipe changes for the selected EoMT model, higher input resolution and four further systems, with the locked environment of the first stage. Every EoMT run used the first stage's budget of 6351 updates. The validation scores below come from models trained on the 497 training images only. [second-stage-runs.csv](../results/second-stage-runs.csv), [second-stage-ensembles.csv](../results/second-stage-ensembles.csv), [class-iou-second-stage.csv](../results/class-iou-second-stage.csv) and [summary-second-stage.json](../results/summary-second-stage.json) hold the full precision values.

### Hidden test result

The submission of 30 October 2026 averaged three H1 models and one H15 model, all retrained on the training and validation images, with scales 1, 1.25 and 1.5 of each input size and horizontal flips. The anonymous [leaderboard API](https://www.codabench.org/api/leaderboards/20330/) returned this snapshot on 30 October 2026 at 11.46 UTC.

| Rank on 30 October 2026 | Participant | Submission | Test mIoU (%) | Test mF1 (%) |
|---:|---|---:|---:|---:|
| 1 | yunusserhat | 1020616 | 61.54 | 71.93 |
| 2 | korayaykor | 1005153 | 60.55 | 71.42 |
| 3 | formid | 1018170 | 47.21 | 58.98 |
| 4 | onurcbayrak | 945126 | 46.02 | 56.99 |
| 5 | MaxwellEQ | 950834 | 33.54 | 40.93 |

The lead over the next entry is 0.98 points of mIoU and 0.51 points of mF1, computed from the full precision values. The counterpart of the submitted ensemble trained only on the training images scored 64.53 mIoU on validation, 2.99 points above the test score, but the two numbers belong to different model instances. An earlier submission of the second stage on 29 October 2026 used the three H1 models alone with scales 0.75, 1 and 1.25. The leaderboard lists only the best entry of each participant and the competition has no published end date, so the ranking is a dated snapshot.

### Single runs

| Run | Input | Seed | Val mIoU (%) | Val mF1 (%) | Best epoch | Updates | Minutes | Peak GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E1 augmentation | 512 by 1024 | 42 | 60.55 | 71.26 | 41 | 6351 | 43.5 | 10.58 |
| E1 augmentation | 512 by 1024 | 123 | 59.12 | 69.76 | 51 | 6351 | 42.9 | 10.58 |
| E1 augmentation | 512 by 1024 | 2026 | 59.10 | 69.54 | 51 | 6351 | 45.9 | 10.58 |
| E2 rare class sampling | 512 by 1024 | 42 | 58.67 | 68.92 | 51 | 6351 | 43.4 | 10.58 |
| E3 augmentation and sampling | 512 by 1024 | 42 | 58.91 | 69.68 | 36 | 6351 | 43.2 | 10.58 |
| P1 four times matching points, stopped | 512 by 1024 | 42 | 58.64 | 68.96 | 36 | 5000 | 34.4 | 10.68 |
| C1 SegFormer-B5, Cityscapes start | 512 by 1024 | 42 | 50.85 | 61.66 | 54 | 7500 | 36.3 | 6.02 |
| C2 Mask2Former Swin-L, longer | 512 by 1024 | 42 | 54.59 | 65.18 | 42 | 7500 | 53.7 | 9.37 |
| C3 YOLO26x semantic, Cityscapes start | 512 by 1024 | 42 | 48.38 | 57.91 | 41 | 10000 | 15.5 | 3.40 |
| C4 EoMT, Cityscapes start | 512 by 1024 | 42 | 57.64 | 68.20 | 36 | 6351 | 49.0 | 10.71 |
| H1 augmentation | 768 by 1536 | 42 | 61.50 | 71.83 | 46 | 6351 | 96.8 | 19.28 |
| H1 augmentation | 768 by 1536 | 123 | 62.29 | 72.08 | 21 | 6351 | 97.0 | 19.29 |
| H1 augmentation | 768 by 1536 | 2026 | 63.56 | 73.24 | 41 | 6351 | 97.2 | 19.29 |
| H1 augmentation | 768 by 1536 | 7 | 61.92 | 72.16 | 31 | 6351 | 96.8 | 19.28 |
| H1 augmentation | 768 by 1536 | 99 | 62.22 | 71.67 | 51 | 6351 | 96.8 | 19.28 |
| H15 augmentation | 896 by 1792 | 42 | 62.71 | 72.24 | 51 | 6351 | 145.4 | 26.14 |

E0, the control, is the first stage's EoMT seed 42 run with 57.97 mIoU. C1 to C4 change the system rather than the EoMT recipe, and C1 to C3 used their own update budgets. After selection, the three H1 recipes and the H15 recipe were retrained on the 581 training and validation images. These runs completed 6351 updates in 44 epochs. Their single validation check at the end gave 80.80 to 82.16 mIoU, which only shows that training ran, because the validation images were part of training.

### Ensembles and test time scales

Each row averages class probabilities with equal weights and horizontal flips. Differences come from the paired image bootstrap against the named baseline. The selection rule was the highest validation mIoU and was written into the scripts before the results existed. The interval describes only the sampling of the 84 images.

| Variant | Scales | Val mIoU (%) | Val mF1 (%) | Baseline | Difference | 95% interval | Positive share |
|---|---|---:|---:|---|---:|---|---:|
| v1-selected-ensemble-remeasured | 0.75 1 1.25 | 60.95 | 71.16 |  |  |  |  |
| e1x3 | 0.75 1 1.25 | 62.04 | 72.46 | v1-selected-ensemble-remeasured | +1.08 | -0.12 to +2.54 | 0.958 |
| h1-seed42 | 0.75 1 1.25 | 62.34 | 72.60 | e1x3 | +0.30 | -0.52 to +2.16 | 0.836 |
| e1x3-h1 | 0.75 1 1.25 | 62.73 | 72.96 | e1x3 | +0.70 | +0.41 to +1.17 | 0.999 |
| e1x3-h1x3 | 0.75 1 1.25 | 63.39 | 73.45 | e1x3-h1 | +0.66 | +0.30 to +1.11 | 1.000 |
| h1x3 | 0.75 1 1.25 | 63.75 | 73.64 | e1x3-h1 | +1.02 | +0.25 to +2.15 | 0.995 |
| h1x3-scales-075-15 | 0.75 1 1.25 1.5 | 63.83 | 73.50 | h1x3 | +0.08 | -0.25 to +0.48 | 0.739 |
| h1x3-scales-05-125 | 0.5 0.75 1 1.25 | 63.57 | 73.66 | h1x3 | -0.18 | -0.69 to +0.35 | 0.235 |
| h1x3-scales-1-15 | 1 1.25 1.5 | 64.01 | 73.48 | h1x3 | +0.26 | -0.50 to +0.99 | 0.768 |
| h1x5 | 1 1.25 1.5 | 64.04 | 73.32 | h1x3-scales-1-15 | +0.02 | -0.64 to +0.76 | 0.549 |
| h1x3-h15 | 1 1.25 1.5 | 64.53 | 73.67 | h1x3-scales-1-15 | +0.52 | +0.02 to +1.02 | 0.979 |
| h1x5-h15 | 1 1.25 1.5 | 64.38 | 73.57 | h1x3-scales-1-15 | +0.37 | -0.20 to +0.98 | 0.880 |
| h1x3-h15-vs-v1 | 1 1.25 1.5 | 64.53 | 73.67 | v1-selected-ensemble-remeasured | +3.58 | +2.31 to +5.59 | 1.000 |

### Negative results

SAM voting in the style of Semantic Segment Anything, a YOLOv8x-seg mask overlay and adding C2 or C3 to an ensemble all lowered validation mIoU. The C3 weights were fixed before its single model result. Rare class sampling (E2) lowered the mean IoU of the eight rarest classes from 31.34 to 30.21 for the same seed. Five H1 seeds gave no clear gain over three.

| Variant | Scales | Val mIoU (%) | Val mF1 (%) | Baseline | Difference | 95% interval | Positive share |
|---|---|---:|---:|---|---:|---|---:|
| ssa-plain | 0.75 1 1.25 | 60.55 | 70.75 | v1-selected-ensemble-remeasured | -0.40 | -1.23 to +0.16 | 0.076 |
| ssa-keep-weak-classes | 0.75 1 1.25 | 60.87 | 71.24 | v1-selected-ensemble-remeasured | -0.09 | -0.83 to +0.39 | 0.289 |
| ssa-skip-small-masks | 0.75 1 1.25 | 60.03 | 70.37 | v1-selected-ensemble-remeasured | -0.92 | -2.43 to +0.06 | 0.032 |
| yolov8-overlay-all-0.5 | 0.75 1 1.25 | 58.92 | 69.49 | e1x3 | -3.12 | -5.53 to -0.33 | 0.010 |
| yolov8-overlay-all-0.7 | 0.75 1 1.25 | 59.44 | 70.20 | e1x3 | -2.60 | -4.51 to -0.03 | 0.022 |
| yolov8-overlay-four-0.5 | 0.75 1 1.25 | 61.39 | 71.94 | e1x3 | -0.65 | -1.65 to -0.08 | 0.013 |
| v1-plus-c3-0.10 | 0.75 1 1.25 | 60.83 | 70.95 |  |  |  |  |
| v1-plus-c3-0.25 | 0.75 1 1.25 | 59.97 | 69.86 |  |  |  |  |
| v1-plus-c3-0.50 | 0.75 1 1.25 | 56.22 | 65.82 |  |  |  |  |
| v1-plus-c2-0.10 | 0.75 1 1.25 | 60.92 | 71.09 | v1-selected-ensemble-remeasured | -0.03 | -0.22 to +0.38 | 0.444 |
| v1-plus-c2-0.25 | 0.75 1 1.25 | 60.20 | 70.21 | v1-selected-ensemble-remeasured | -0.75 | -1.36 to +0.17 | 0.070 |
| v1-plus-c2-0.50 | 0.75 1 1.25 | 58.66 | 68.31 | v1-selected-ensemble-remeasured | -2.29 | -3.42 to -0.59 | 0.002 |
| e2-plus-c2-equal | 0.75 1 1.25 | 58.90 | 68.67 | v1-selected-ensemble-remeasured | -2.06 | -3.24 to -0.15 | 0.015 |

### Compute and limits

The 20 training runs of this stage took 24.56 GPU hours of elapsed process time, including validation. This excludes calibration probes, the YOLOv8 overlay training, SAM mask generation and inference. The test prediction with four models took 1070.92 seconds for 249 images, or 1088.71 seconds including checkpoint loading, with 5.63 GiB of peak allocated memory. The first stage needed 242.19 and 251.49 seconds and 2.70 GiB.

The 84 validation images selected checkpoints, test time scales and the composition, so the validation scores are optimistic. The submitted models have no independent validation score. The second round of runs was approved right after the first submission of the second stage, with the aim of passing the leading entry. Its choices used only validation data and the written rules, and no setting was tuned on the test set. Paired bootstrap intervals ignore the correlation of nearby views and the selection process.

## First stage, 5 October 2026

The result on the public Codabench leaderboard was verified through its [anonymous API](https://www.codabench.org/api/leaderboards/20330/) on 5 October 2026. Submission **961441** by **yunusserhat** obtained **57.08 mIoU** and **67.96 mF1**, ranking first among the three entries in that snapshot. The submission was recorded at 06:15 UTC, or 09:15 in Istanbul. The lead over the next entry was 11.05 percentage points of mIoU. This is a dated leaderboard result rather than a claim about all future submissions.

| Rank on 5 October 2026 | Participant | Submission | Test mIoU (%) | Test mF1 (%) |
|---:|---|---:|---:|---:|
| 1 | yunusserhat | 961441 | 57.08 | 67.96 |
| 2 | onurcbayrak | 945126 | 46.02 | 56.99 |
| 3 | MaxwellEQ | 950834 | 33.54 | 40.93 |

The chosen predictor averages the class probabilities of three independently finetuned EoMT DINOv3 ViT Large models, using seeds 42, 123 and 2026. Each image uses scales 0.75, 1 and 1.25 with horizontal flip. Models are placed on the GPU one at a time. The full prediction is resized to the original 1024 by 512 panorama before class selection.

The validation split contains 84 images and was held out from training, but reused for checkpoint, seed and inference selection. Its best score is not an independent test estimate. The hidden test score above was obtained once by uploading the selected ZIP. Hidden test labels were never accessed and no leaderboard based tuning was used.

### Pilot comparison

Nine candidate systems received approximately 600 seconds of training plus validation compute each, with seed 42 and ten planned validation checks. The measured totals range from 603.15 to 628.02 seconds. Four fixed source recipe corrections replaced their original recipes in the primary comparison. The original and corrected scores were not maximized against one another. All 13 pilot runs remain in the measurement file.

| Candidate | Val mIoU (%) | Val mF1 (%) | Train plus val (s) |
|---|---:|---:|---:|
| eomt_dinov3_vit_large | 57.57 | 68.52 | 624.20 |
| dinov3_vit_large_linear | 52.15 | 63.57 | 628.02 |
| mask2former_swin_large_source_optimizer | 51.93 | 62.88 | 609.14 |
| dinov3_vit_base_linear | 49.11 | 60.30 | 603.15 |
| upernet_convnext_large_source_scalar | 48.78 | 59.04 | 609.52 |
| upernet_swin_large | 46.88 | 56.32 | 605.99 |
| segformer_mit_b2_headlr10 | 46.22 | 56.78 | 608.80 |
| segformer_mit_b5_transformers_headlr10 | 45.74 | 55.96 | 618.96 |
| deeplabv3plus_resnet50 | 42.31 | 51.54 | 619.87 |

### Confirmation and inference

The two pilot leaders each received approximately 2400 seconds per seed, with seeds 42, 123 and 2026 and ten planned validation checks. EoMT DINOv3 Large averaged **58.79 ± 0.70 mIoU**, compared with **54.49 ± 0.28** for the DINOv3 Large linear decoder. These are population standard deviations in percentage points across three seeds, not confidence intervals.

| Inference variant | Val mIoU (%) | Val mF1 (%) | Prediction for 84 images (s) | Allocated VRAM (GiB) |
|---|---:|---:|---:|---:|
| selected-ensemble-multiscale-flip | 60.95 | 71.16 | 81.72 | 2.70 |
| winner-multiscale-flip | 60.67 | 70.83 | 16.77 | 2.70 |
| winner-seeds-equal | 59.85 | 70.09 | 51.82 | 2.32 |
| winner-single | 59.67 | 69.72 | 7.85 | 2.32 |
| cross-winner75 | 59.65 | 69.62 | 34.72 | 2.32 |
| selected-ensemble-flip | 59.51 | 69.59 | 56.95 | 2.39 |
| winner-window | 59.07 | 69.81 | 13.68 | 1.91 |
| winner-flip | 59.03 | 68.93 | 8.34 | 2.39 |
| cross-equal | 58.36 | 68.72 | 34.78 | 2.32 |
| cross-runner75 | 55.86 | 66.84 | 34.69 | 2.32 |
| runner-single | 54.71 | 65.86 | 6.96 | 1.79 |

The selected ensemble achieved **60.95 mIoU** and **71.16 mF1** on validation, improving mIoU by **1.28 percentage points** over the best native single model. A single EoMT model with the same multiscale and flip procedure achieved **60.67 mIoU** in 16.77 seconds for 84 images. The selected ensemble needed 81.72 seconds and 2.70 GiB of peak allocated VRAM. The additional 0.28 points cost 4.87 times as much prediction time. The highest validation mIoU was chosen because its final test runtime remained practical.

The 249 image test prediction took **242.19 seconds**, or **251.49 seconds** including checkpoint loading and provenance hashes. Peak allocated and reserved CUDA memory were **2.70** and **3.61 GiB**. Timings were measured on an RTX 5090 with BF16. Prediction timing includes decode, preprocessing, CPU and GPU transfers, forward passes, argmax and PNG writes.

### Verification and scope

The public scorer computes a single dataset confusion matrix and the macro mean over all 32 classes, including Void. Classes absent from both truth and prediction contribute zero. An independent audit decoded all **924 validation PNGs** from the eleven variants and obtained the recorded confusion matrices and exact public scorer metrics. The training audit covered **19 runs and 190 validation checks**. The submission audit verified all **249** names, PNG modes, dimensions, class ranges and member hashes.

The four CSV files and [summary.json](../results/summary.json) contain full precision measurements, class IoUs, seed results, all nineteen training runs and the dated public leaderboard snapshot. Metric values inside the JSON are fractions unless explicitly labelled percent. The recorded experiment code SHA was `3d3e77d215319916822958b3743d66ed65fc5399d5a1d5a76dfffde99adb6af2`. The portable release adapts workspace, setup and pretrained provenance paths and therefore has a different code identity. It has not been remeasured as a new training experiment.

Exact pixel duplicates between splits were absent, but nearby train and validation views were observed. Overpass and Pruned Tree have only 40 and 54 truth pixels in validation, each appearing in one image. Both remain at zero IoU. Five classes regress relative to the native single model, including Bus and Motorcycle. Pretraining data and initial decoder weights differ across models, so the pilot is a practical transfer comparison rather than an architecture controlled ablation.

The original selected submission ZIP SHA256 is `c0227ed7c6b9d4a718ba551598165df4d1439cecfd7aac659c3674bd1275bf26`. The dataset, checkpoints and ZIP are not distributed in this code repository. The workflow recreates training and prediction from the official data and pinned upstream sources. CUDA deterministic mode was disabled, so an independently retrained model is not guaranteed to reproduce identical weights or the exact leaderboard score.
