#!/bin/bash

set -e

DATASET="DocVQA_VAL"
TIME="03:00:00"

# ============================================================
# Model Zoo
# ============================================================

MODELS=(
    "NEO-2B-SFT" # original model
    
    "NEO-2B-SFT-Fixed-2x"
    "NEO-2B-SFT-Fixed-4x"
    "NEO-2B-SFT-Fixed-8x" # fixed pooling

    "NEO-2B-SFT-Random-2x"
    "NEO-2B-SFT-Random-4x"
    "NEO-2B-SFT-Random-8x" # random pooling

    "NEO-2B-SFT-Spatial-4x" 
    "NEO-2B-SFT-Spatial-8x" # spatial pooling

    "NEO-2B-SFT-ToMe-2x"
    "NEO-2B-SFT-ToMe-4x"
    "NEO-2B-SFT-ToMe-8x" # ToMe

    "NEO-2B-SFT-PruneSID-2x"
    "NEO-2B-SFT-PruneSID-4x"
    "NEO-2B-SFT-PruneSID-8x" # PruneSID

    "NEO-2B-SFT-CDPruner-2x"
    "NEO-2B-SFT-CDPruner-4x"
    "NEO-2B-SFT-CDPruner-8x" # CDPruner

    "NEO-2B-SFT-PDrop-layer16-2x"
    "NEO-2B-SFT-PDrop-layer16-4x"
    "NEO-2B-SFT-PDrop-layer16-8x" # PDrop

    "NEO-2B-SFT-FastV-layer2-2x"
    "NEO-2B-SFT-FastV-layer2-4x"
    "NEO-2B-SFT-FastV-layer2-8x" # FastV


    # "NEO-2B-SFT-OV-LLaVA-llava66k-DRIP-2x-two-tailed"
    # "NEO-2B-SFT-OV-LLaVA-llava66k-DRIP-4x-two-tailed"
    # "NEO-2B-SFT-OV-LLaVA-llava66k-DRIP-8x-two-tailed" # VLVLM (two-tailed design)


    # "NEO-2B-SFT-OV-LLaVA-llava66k-Perceiver-2x"
    # "NEO-2B-SFT-OV-LLaVA-llava66k-Perceiver-4x"
    # "NEO-2B-SFT-OV-LLaVA-llava66k-Perceiver-8x" # Perceiver
)

mkdir -p logs

# ============================================================
# Submit evaluation jobs
# ============================================================

submit_eval() {
    local MODEL=$1

    sbatch \
        --job-name="DocVQA_VAL_${MODEL}" \
        --time="$TIME" \
        --output="logs/${MODEL}_DocVQA_VAL_%j.out" \
        /users/PAS2912/yusenpeng/VTC-Eval/VTCeval_scripts/eval.sbatch \
        "$DATASET" \
        "$MODEL"
}

for MODEL in "${MODELS[@]}"; do
    echo "Submitting DocVQA_VAL evaluation for ${MODEL}..."
    submit_eval "$MODEL"
done

echo "All DocVQA_VAL jobs submitted!"
