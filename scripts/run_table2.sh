#!/bin/bash
# Run on an OSC login node from the repo root on scratch:
#   cd /fs/scratch/PAS2836/$USER/VTC-Native-Eval && git pull --ff-only && bash scripts/run_table2.sh ocr
# Submits the 24-task array for one suite, then (optionally) waits for it and prints the Table 2 summary.
#   bash scripts/run_table2.sh [general|ocr]          # submit and return; summarize later by hand
#   bash scripts/run_table2.sh [general|ocr] --wait   # block until all tasks finish, then summarize (use inside tmux)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

SUITE=general; WAIT=()
for arg in "$@"; do
    case "$arg" in
        general|ocr) SUITE=$arg ;;
        --wait) WAIT=(--wait) ;;
        *) echo "usage: $0 [general|ocr] [--wait]"; exit 2 ;;
    esac
done
# walltime limit per task: general <= 25 min observed; ocr: DocVQA ~2 h at full resolution
case "$SUITE" in
    general) TIME=1:00:00 ;;
    ocr)     TIME=6:00:00 ;;
esac

jid=$(sbatch --parsable "${WAIT[@]}" --time="$TIME" --job-name="table2_$SUITE" \
      --export=ALL,SUITE="$SUITE" scripts/eval_table2.sbatch)

if [[ ${#WAIT[@]} -gt 0 ]]; then
    echo "array $jid finished"
    sacct -j "$jid" -X --format=JobID%14,JobName%14,Elapsed,State,ExitCode
    module load miniconda3/24.1.2-py310 && source activate "/fs/scratch/PAS2836/$USER/conda/envs/neo"
    python scripts/summarize_table2.py --latex
else
    echo "submitted array $jid (24 tasks, suite=$SUITE, limit $TIME each)."
    echo "Watch: squeue --me    Logs: logs/table2_${SUITE}_${jid}_*.log"
    echo "When done:  python scripts/summarize_table2.py --latex"
fi
