#!/bin/bash

set -e

MODEL=${1:-NEO-2B-SFT}

mkdir -p logs

submit_eval() {
    local NAME=$1
    local DATASET=$2
    local TIME=$3

    sbatch \
        --job-name="$NAME" \
        --time="$TIME" \
        --output="${MODEL}_${NAME}_%j.out" \
        /users/PAS2912/yusenpeng/VTC-Eval/VTCeval_scripts/eval.sbatch \
        "$DATASET" \
        "$MODEL"
}

submit_eval "MMBench"     "MMBench_DEV_EN"  "00:30:00"
submit_eval "MME"         "MME"             "00:30:00"
submit_eval "MMMU"        "MMMU_DEV_VAL"    "00:30:00"
submit_eval "RealWorldQA" "RealWorldQA"     "00:20:00"
submit_eval "TextVQA"     "TextVQA_VAL"     "01:20:00"
submit_eval "DocVQA"      "DocVQA_VAL"      "02:20:00"
submit_eval "OCRBench"    "OCRBench"        "00:25:00"
submit_eval "ChartQA"     "ChartQA_TEST"    "00:45:00"