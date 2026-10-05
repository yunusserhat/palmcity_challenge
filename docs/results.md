# Recorded challenge results

The result on the public Codabench leaderboard was verified through its [anonymous API](https://www.codabench.org/api/leaderboards/20330/) on 5 October 2026. Submission **961441** by **yunusserhat** obtained **57.08 mIoU** and **67.96 mF1**, ranking first among the three entries in that snapshot. The submission was recorded at 06:15 UTC, or 09:15 in Istanbul. The lead over the next entry was 11.05 percentage points of mIoU. This is a dated leaderboard result rather than a claim about all future submissions.

| Rank on 5 October 2026 | Participant | Submission | Test mIoU (%) | Test mF1 (%) |
|---:|---|---:|---:|---:|
| 1 | yunusserhat | 961441 | 57.08 | 67.96 |
| 2 | onurcbayrak | 945126 | 46.02 | 56.99 |
| 3 | MaxwellEQ | 950834 | 33.54 | 40.93 |

The chosen predictor averages the class probabilities of three independently finetuned EoMT DINOv3 ViT Large models, using seeds 42, 123 and 2026. Each image uses scales 0.75, 1 and 1.25 with horizontal flip. Models are placed on the GPU one at a time. The full prediction is resized to the original 1024 by 512 panorama before class selection.

The validation split contains 84 images and was held out from training, but reused for checkpoint, seed and inference selection. Its best score is not an independent test estimate. The hidden test score above was obtained once by uploading the selected ZIP. Hidden test labels were never accessed and no leaderboard based tuning was used.

## Pilot comparison

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

## Confirmation and inference

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

## Verification and scope

The public scorer computes a single dataset confusion matrix and the macro mean over all 32 classes, including Void. Classes absent from both truth and prediction contribute zero. An independent audit decoded all **924 validation PNGs** from the eleven variants and obtained the recorded confusion matrices and exact public scorer metrics. The training audit covered **19 runs and 190 validation checks**. The submission audit verified all **249** names, PNG modes, dimensions, class ranges and member hashes.

The four CSV files and [summary.json](../results/summary.json) contain full precision measurements, class IoUs, seed results, all nineteen training runs and the dated public leaderboard snapshot. Metric values inside the JSON are fractions unless explicitly labelled percent. The recorded experiment code SHA was `3d3e77d215319916822958b3743d66ed65fc5399d5a1d5a76dfffde99adb6af2`. The portable release adapts workspace, setup and pretrained provenance paths and therefore has a different code identity. It has not been remeasured as a new training experiment.

Exact pixel duplicates between splits were absent, but nearby train and validation views were observed. Overpass and Pruned Tree have only 40 and 54 truth pixels in validation, each appearing in one image. Both remain at zero IoU. Five classes regress relative to the native single model, including Bus and Motorcycle. Pretraining data and initial decoder weights differ across models, so the pilot is a practical transfer comparison rather than an architecture controlled ablation.

The original selected submission ZIP SHA256 is `c0227ed7c6b9d4a718ba551598165df4d1439cecfd7aac659c3674bd1275bf26`. The dataset, checkpoints and ZIP are not distributed in this code repository. The workflow recreates training and prediction from the official data and pinned upstream sources. CUDA deterministic mode was disabled, so an independently retrained model is not guaranteed to reproduce identical weights or the exact leaderboard score.
