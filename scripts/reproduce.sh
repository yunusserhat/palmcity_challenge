#!/usr/bin/env bash
# Reproduce the selected three-seed EoMT recipe on a new, explicit workspace.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$PALMCITY_CODE_ROOT"
stage="${1:-help}"
device="${PALMCITY_DEVICE:-cuda:0}"
manifest="$PALMCITY_WORKSPACE/manifests/dataset.json"
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
    *)
        printf '%s\n' 'Usage: bash scripts/reproduce.sh {data|prepare [--download]|train|val|test}'
        [[ "$stage" == help ]] || exit 2
        ;;
esac
