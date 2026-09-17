#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=neo2b_eval
#SBATCH --partition=nextgen
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%A_%a.log
#
# One array task per benchmark (per-dataset failure isolation, runs in parallel).
# Submit from the repo root on scratch:  mkdir -p logs && sbatch eval_neo2b.sbatch
# Single dataset:                        sbatch --array=3 eval_neo2b.sbatch
set -uo pipefail

DATASETS=(MMBench_DEV_EN MME MMMU_DEV_VAL RealWorldQA GQA_TestDev_Balanced
          TextVQA_VAL OCRBench ChartQA_TEST DocVQA_VAL OCRBench_v2)
DATA=${DATASETS[$SLURM_ARRAY_TASK_ID]}
MODEL=${MODEL:-NEO-2B-SFT}

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
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"

python run.py --data "$DATA" --model "$MODEL" --verbose
rc=$?
echo "===== $(date) :: $DATA finished rc=$rc ====="
exit $rc
