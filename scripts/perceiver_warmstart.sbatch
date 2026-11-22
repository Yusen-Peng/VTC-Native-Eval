#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=pl_warmstart
#SBATCH --partition=debug-nextgen
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.log
#
# scripts/perceiver_warmstart.py on one GPU (NEO-2B vision embedding on N LLaVA images, ~5-10 min). Submit from the
# checkout root (mkdir -p logs):
#   REPO=$PWD RATIO=2 N=256 sbatch --export=ALL scripts/perceiver_warmstart.sbatch
# Output: $OUT (default $SCRATCH/VTC-Native-Eval-ckpts/init/perceiver_learned_warm_<RATIO>x.bin) + .json sidecar.
set -euo pipefail

module reset
module load miniconda3/24.1.2-py310
export SCRATCH=/fs/scratch/PAS2836/$USER
source activate "$SCRATCH/conda/envs/neo"

unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE
export HF_HOME=$SCRATCH/hf
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO=${REPO:-$SCRATCH/VTC-Native-Eval}
RATIO=${RATIO:-2}
OUT=${OUT:-$SCRATCH/VTC-Native-Eval-ckpts/init/perceiver_learned_warm_${RATIO}x.bin}
mkdir -p "$(dirname "$OUT")"

echo "===== $(date) :: perceiver warm start, ratio ${RATIO}x, N=${N:-256}, repo $REPO ====="
echo "commit: $(git -C "$REPO" rev-parse HEAD)"
nvidia-smi -L

cd "$TMPDIR"
python "$REPO/scripts/perceiver_warmstart.py" --ratio "$RATIO" --n "${N:-256}" --out "$OUT" ${EXTRA:-}
rc=$?
echo "===== $(date) :: finished rc=$rc, wrote $OUT ====="
exit $rc
