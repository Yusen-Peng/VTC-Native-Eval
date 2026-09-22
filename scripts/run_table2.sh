#!/bin/bash
# Run on an OSC login node from the repo root on scratch:
#   cd /fs/scratch/PAS2836/$USER/VTC-Native-Eval && git pull --ff-only && bash scripts/run_table2.sh
# Submits the 24-task array, then (optionally) waits for it and prints the Table 2 summary.
#   bash scripts/run_table2.sh          # submit and return; summarize later by hand
#   bash scripts/run_table2.sh --wait   # block until all tasks finish, then summarize (use inside tmux)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

if [[ "${1:-}" == "--wait" ]]; then
    jid=$(sbatch --parsable --wait scripts/eval_table2.sbatch)
    echo "array $jid finished"
    sacct -j "$jid" -X --format=JobID%14,JobName%10,Elapsed,State,ExitCode
    module load miniconda3/24.1.2-py310 && source activate "/fs/scratch/PAS2836/$USER/conda/envs/neo"
    python scripts/summarize_table2.py --latex
else
    jid=$(sbatch --parsable scripts/eval_table2.sbatch)
    echo "submitted array $jid (24 tasks). Watch: squeue --me    Logs: logs/table2_${jid}_*.log"
    echo "When done:  python scripts/summarize_table2.py --latex"
fi
