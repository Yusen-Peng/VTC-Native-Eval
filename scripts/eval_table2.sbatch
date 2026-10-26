#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=table2
#SBATCH --partition=nextgen
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --array=4-59
#SBATCH --output=logs/%x_%A_%a.log
#
# Table 2 (training-free) at one SCALE (2B | 9B): baseline NEO-<SCALE>-SFT + {fixed, random, tome, prunesid} x {2x, 4x, 8x}
# + spatial x {4x, 8x} -> 15 models, x 4 benchmarks of one SUITE -> 60 tasks, each on its own GPU, all in parallel.
#   SUITE=general (default): MMBench, MME, MMMU, RealWorldQA
#   SUITE=ocr:               TextVQA, DocVQA, OCRBench, ChartQA   (DocVQA ~2 h at 2B; submit with a longer --time)
# idx = model_idx*4 + dataset_idx: base = 0-3, fixed = 4-15, random = 16-27, tome = 28-39, prunesid = 40-51, spatial = 52-59
# Submit via scripts/run_table2.sh, which sets SCALE / SUITE / --time / --job-name / --array.
# Re-run one task, resuming its predictions (REUSE=1 passes --reuse to run.py; without it run.py starts over):
#   sbatch --array=<idx> --time=<t> --export=ALL,SCALE=9B,SUITE=ocr,REUSE=1 scripts/eval_table2.sbatch
# Other models (e.g. trained checkpoints): MODELS="<config.py keys>" replaces the variant list, idx = model_idx*4 + dataset_idx:
#   sbatch --array=0-11 --time=3:00:00 --export=ALL,SUITE=general,MODELS="m1 m2 m3" scripts/eval_table2.sbatch
# Out of GPU memory (e.g. 9B baseline on multi-image MMMU, idx 2): same, plus --gpus-per-node=2 (device_map="auto" splits layers).
set -uo pipefail

SCALE=${SCALE:-2B}
VARIANTS=("" -Fixed-2x -Fixed-4x -Fixed-8x -Random-2x -Random-4x -Random-8x -ToMe-2x -ToMe-4x -ToMe-8x
          -PruneSID-2x -PruneSID-4x -PruneSID-8x -Spatial-4x -Spatial-8x)
SUITE=${SUITE:-general}
case "$SUITE" in
    general) DATASETS=(MMBench_DEV_EN MME MMMU_DEV_VAL RealWorldQA) ;;
    ocr)     DATASETS=(TextVQA_VAL DocVQA_VAL OCRBench ChartQA_TEST) ;;
    *) echo "unknown SUITE=$SUITE (expected general|ocr)"; exit 2 ;;
esac
if [ -n "${MODELS:-}" ]; then
    # MODELS="name1 name2 ..." (config.py keys, e.g. trained checkpoints) replaces the variant list
    read -ra MODEL_LIST <<< "$MODELS"
    MODEL=${MODEL_LIST[$((SLURM_ARRAY_TASK_ID / ${#DATASETS[@]}))]}
else
    MODEL=NEO-$SCALE-SFT${VARIANTS[$((SLURM_ARRAY_TASK_ID / ${#DATASETS[@]}))]}
fi
DATA=${DATASETS[$((SLURM_ARRAY_TASK_ID % ${#DATASETS[@]}))]}

module reset
module load miniconda3/24.1.2-py310
export SCRATCH=/fs/scratch/PAS2836/$USER
source activate "$SCRATCH/conda/envs/neo"

unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE   # legacy vars would override HF_HOME
export HF_HOME=$SCRATCH/hf
export HF_HUB_OFFLINE=1          # weights are pre-cached; never hit the network from a compute node
export LMUData=$SCRATCH/LMUData  # must be user-writable (VLMEvalKit decodes images into it)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # long eager-attention prefills fragment memory

REPO=$SCRATCH/VTC-Native-Eval
cd "$REPO/VLMEvalKit"

echo "===== $(date) :: $MODEL / $DATA ====="
echo "commit: $(git -C "$REPO" rev-parse HEAD)"
nvidia-smi -L

python run.py --data "$DATA" --model "$MODEL" --verbose ${REUSE:+--reuse}
rc=$?
echo "===== $(date) :: $MODEL / $DATA finished rc=$rc ====="
exit $rc
