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
#SBATCH --array=0-23
#SBATCH --output=logs/%x_%A_%a.log
#
# Table 2 (training-free): {fixed, random} x {2x, 4x, 8x} x {MMBench, MME, MMMU, RealWorldQA}.
# One array task per (model, dataset) pair -> 24 tasks, each on its own GPU, all in parallel.
# Submit via scripts/run_table2.sh (or directly: mkdir -p logs && sbatch scripts/eval_table2.sbatch).
# Re-run one pair:  sbatch --array=<idx> scripts/eval_table2.sbatch   (idx = model_idx*4 + dataset_idx)
set -uo pipefail

MODELS=(NEO-2B-SFT-Fixed-2x NEO-2B-SFT-Fixed-4x NEO-2B-SFT-Fixed-8x
        NEO-2B-SFT-Random-2x NEO-2B-SFT-Random-4x NEO-2B-SFT-Random-8x)
DATASETS=(MMBench_DEV_EN MME MMMU_DEV_VAL RealWorldQA)
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
