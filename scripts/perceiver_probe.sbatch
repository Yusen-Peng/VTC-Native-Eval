#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=pl_probe
#SBATCH --partition=debug-nextgen
#SBATCH --time=0:50:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.log
#
# scripts/perceiver_probe.py on one GPU (NEO-2B, ~15-20 min for N=128). Submit from the checkout root (mkdir -p logs):
#   REPO=$PWD sbatch --export=ALL,REPO,N=128 scripts/perceiver_probe.sbatch
# Results: $OUT (default $SCRATCH/VTC-Native-Eval-ckpts/probes/perceiver_probe_<jobid>.json) and the log.
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
OUT=${OUT:-$SCRATCH/VTC-Native-Eval-ckpts/probes/perceiver_probe_${SLURM_JOB_ID}.json}
mkdir -p "$(dirname "$OUT")"

echo "===== $(date) :: perceiver probe, N=${N:-128}, repo $REPO ====="
echo "commit: $(git -C "$REPO" rev-parse HEAD)"
nvidia-smi -L

cd "$TMPDIR"
python "$REPO/scripts/perceiver_probe.py" --out "$OUT" --n "${N:-128}" ${EXTRA:-}
rc=$?
echo "===== $(date) :: finished rc=$rc, results $OUT ====="
exit $rc
