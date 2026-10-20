#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --account=PAS2836
#SBATCH --partition=nextgen

set -e

DATASET=$1
MODEL=${2:-NEO-2B-SFT}

module load miniconda3/24.1.2-py310
conda activate neo
export LMUData=/fs/scratch/PAS2836/yusenpeng_dataset/VLMEvalKit_data

cd "$SLURM_SUBMIT_DIR"

echo "Job: $SLURM_JOB_NAME"
echo "Dataset: $DATASET"
echo "Model: $MODEL"

cd VLMEvalKit

python run.py \
    --data "$DATASET" \
    --model "$MODEL" \
    --verbose

cd ..