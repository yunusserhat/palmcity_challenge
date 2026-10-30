#!/usr/bin/env bash
# Reproduce the selected systems on a new, explicit workspace.
# Stages without a prefix rebuild the 5 October 2026 system, three EoMT seeds at 512x1024.
# v2 stages rebuild the 30 October 2026 system, three H1 seeds at 768x1536 and one H15 model at 896x1792.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$PALMCITY_CODE_ROOT"
stage="${1:-help}"
device="${PALMCITY_DEVICE:-cuda:0}"
manifest="$PALMCITY_WORKSPACE/manifests/dataset.json"
trainval_manifest="$PALMCITY_WORKSPACE/manifests/trainval.json"
v2_selected=(campaign-h1-augment-768x1536-seed42 campaign-h1-augment-768x1536-seed123
             campaign-h1-augment-768x1536-seed2026 campaign-h15-augment-896x1792-seed42)
v2_final=(final-h1-trainval-seed42 final-h1-trainval-seed123 final-h1-trainval-seed2026 final-h15-trainval-seed42)
v2_inference=(--weights 1 1 1 1 --scales 1 1.25 1.5 --hflip-tta --max-scaled-pixels 4000000)
train_runs() {
    local run_manifest="$1" run_name
    shift
    for run_name in "$@"; do
        local resume_args=()
        if [[ -f "$PALMCITY_WORKSPACE/runs/$run_name/latest.pt" ]]; then
            resume_args=(--resume "$PALMCITY_WORKSPACE/runs/$run_name/latest.pt")
        fi
        bash scripts/run.sh python -m palmcity.train \
            --config "$PALMCITY_WORKSPACE/configs/$run_name.json" \
            --manifest "$run_manifest" --run-name "$run_name" --device "$device" \
            --allow-pretrained-downloads "${resume_args[@]}"
    done
}
best_checkpoints() {
    local run_name
    for run_name in "$@"; do
        printf '%s\n' "$PALMCITY_WORKSPACE/runs/$run_name/best.pt"
    done
}
case "$stage" in
    data)
        bash scripts/run.sh python -m palmcity.download_data
        bash scripts/run.sh python -m palmcity.data build \
            --root "$PALMCITY_WORKSPACE/data/PalmCity" --output "$manifest" \
            --report reports/dataset-audit.json --strict-counts
        ;;
    prepare)
        for seed in 42 123 2026; do
            bash scripts/run.sh python -m palmcity.pretrained \
                --recipe "configs/recipes/confirmation-eomt_dinov3_vit_large-seed${seed}.json" \
                --output "configs/eomt-seed${seed}.json" "${@:2}"
        done
        ;;
    train)
        bash scripts/run.sh python -m palmcity.storage --min-free-gib 50
        for seed in 42 123 2026; do
            run_name="confirmation-eomt_dinov3_vit_large-seed${seed}"
            resume_args=()
            if [[ -f "$PALMCITY_WORKSPACE/runs/$run_name/latest.pt" ]]; then
                resume_args=(--resume "$PALMCITY_WORKSPACE/runs/$run_name/latest.pt")
            fi
            bash scripts/run.sh python -m palmcity.train \
                --config "$PALMCITY_WORKSPACE/configs/eomt-seed${seed}.json" \
                --manifest "$manifest" --run-name "$run_name" --device "$device" \
                --allow-pretrained-downloads "${resume_args[@]}"
        done
        ;;
    val|test)
        checkpoints=()
        for seed in 42 123 2026; do
            checkpoints+=("$PALMCITY_WORKSPACE/runs/confirmation-eomt_dinov3_vit_large-seed${seed}/best.pt")
        done
        bash scripts/run.sh python -m palmcity.predict --checkpoint "${checkpoints[@]}" \
            --weights 1 1 1 --manifest "$manifest" --output-dir "outputs/$stage-ensemble-tta" \
            --split "$stage" --device "$device" --scales 0.75 1 1.25 --hflip-tta
        if [[ "$stage" == val ]]; then
            bash scripts/run.sh python -m palmcity.evaluate --manifest "$manifest" \
                --predictions "$PALMCITY_WORKSPACE/outputs/val-ensemble-tta" \
                --report reports/val-ensemble-tta-score.json
        else
            bash scripts/run.sh python -m palmcity.submission --manifest "$manifest" \
                --predictions "$PALMCITY_WORKSPACE/outputs/test-ensemble-tta" \
                --output outputs/submission.zip
        fi
        ;;
    v2-prepare)
        for run_name in "${v2_selected[@]}" "${v2_final[@]}"; do
            bash scripts/run.sh python -m palmcity.pretrained \
                --recipe "configs/recipes/$run_name.json" \
                --output "configs/$run_name.json" "${@:2}"
        done
        ;;
    v2-train)
        bash scripts/run.sh python -m palmcity.storage --min-free-gib 50
        train_runs "$manifest" "${v2_selected[@]}"
        ;;
    v2-val)
        mapfile -t checkpoints < <(best_checkpoints "${v2_selected[@]}")
        bash scripts/run.sh python -m palmcity.predict --checkpoint "${checkpoints[@]}" "${v2_inference[@]}" \
            --manifest "$manifest" --output-dir outputs/v2-val-ensemble --split val --device "$device"
        bash scripts/run.sh python -m palmcity.evaluate --manifest "$manifest" \
            --predictions "$PALMCITY_WORKSPACE/outputs/v2-val-ensemble" \
            --report reports/v2-val-ensemble-score.json
        ;;
    v2-trainval)
        if [[ ! -f "$trainval_manifest" ]]; then
            bash scripts/run.sh python -m palmcity.data trainval --manifest "$manifest" \
                --output "$trainval_manifest" --strict-counts
        fi
        bash scripts/run.sh python -m palmcity.storage --min-free-gib 50
        train_runs "$trainval_manifest" "${v2_final[@]}"
        ;;
    v2-test)
        mapfile -t checkpoints < <(best_checkpoints "${v2_final[@]}")
        bash scripts/run.sh python -m palmcity.predict --checkpoint "${checkpoints[@]}" "${v2_inference[@]}" \
            --manifest "$manifest" --output-dir outputs/v2-test-ensemble --split test --device "$device"
        bash scripts/run.sh python -m palmcity.submission --manifest "$manifest" \
            --predictions "$PALMCITY_WORKSPACE/outputs/v2-test-ensemble" \
            --output outputs/v2-submission.zip
        ;;
    *)
        printf '%s\n' 'Usage: bash scripts/reproduce.sh {data|prepare [--download]|train|val|test|v2-prepare [--download]|v2-train|v2-val|v2-trainval|v2-test}'
        [[ "$stage" == help ]] || exit 2
        ;;
esac
