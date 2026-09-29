#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=table2
#SBATCH --partition=nextgen
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --array=0-35
#SBATCH --output=logs/%x_%A_%a.log
#
# Table 2 (training-free): {fixed, random, tome} x {2x, 4x, 8x} x 4 benchmarks of one SUITE:
#   SUITE=general (default): MMBench, MME, MMMU, RealWorldQA      (<= 25 min per task)
#   SUITE=ocr:               TextVQA, DocVQA, OCRBench, ChartQA   (up to ~2 h per task; submit with --time=6:00:00)
# One array task per (model, dataset) pair -> 36 tasks, each on its own GPU, all in parallel.
# Submit via scripts/run_table2.sh [general|ocr] [fixed|random|tome], which sets SUITE / --time / --job-name / --array.
# Re-run one pair:  sbatch --array=<idx> --export=ALL,SUITE=ocr --time=6:00:00 scripts/eval_table2.sbatch
#                   (idx = model_idx*4 + dataset_idx; fixed = 0-11, random = 12-23, tome = 24-35)
set -uo pipefail

MODELS=(NEO-2B-SFT-Fixed-2x NEO-2B-SFT-Fixed-4x NEO-2B-SFT-Fixed-8x
        NEO-2B-SFT-Random-2x NEO-2B-SFT-Random-4x NEO-2B-SFT-Random-8x
        NEO-2B-SFT-ToMe-2x NEO-2B-SFT-ToMe-4x NEO-2B-SFT-ToMe-8x)
SUITE=${SUITE:-general}
case "$SUITE" in
    general) DATASETS=(MMBench_DEV_EN MME MMMU_DEV_VAL RealWorldQA) ;;
    ocr)     DATASETS=(TextVQA_VAL DocVQA_VAL OCRBench ChartQA_TEST) ;;
    *) echo "unknown SUITE=$SUITE (expected general|ocr)"; exit 2 ;;
esac
MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID / ${#DATASETS[@]}))]}
DATA=${DATASETS[$((SLURM_ARRAY_TASK_ID % ${#DATASETS[@]}))]}

module reset
module load miniconda3/24.1.2-py310
export SCRATCH=/fs/scratch/PAS2836/$USER
source activate "$SCRATCH/conda/envs/neo"

unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE   # legacy vars would override HF_HOME
export HF_HOME=$SCRATCH/hf
export HF_HUB_OFFLINE=1          # weights are pre-cached; never hit the network from a compute node
export LMUData=$SCRATCH/LMUData  # must be user-writable (VLMEvalKit decodes images into it)

REPO=$SCRATCH/VTC-Native-Eval
cd "$REPO/VLMEvalKit"

echo "===== $(date) :: $MODEL / $DATA ====="
echo "commit: $(git -C "$REPO" rev-parse HEAD)"
nvidia-smi -L

python run.py --data "$DATA" --model "$MODEL" --verbose
rc=$?
echo "===== $(date) :: $MODEL / $DATA finished rc=$rc ====="
exit $rc
