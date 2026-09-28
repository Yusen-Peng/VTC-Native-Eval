#!/bin/bash
# Run on an OSC login node from the repo root on scratch:
#   cd /fs/scratch/PAS2836/$USER/VTC-Native-Eval && git pull --ff-only && bash scripts/run_table2.sh
# Submits one task array per suite, then (optionally) waits for them and prints the Table 2 summary.
#   bash scripts/run_table2.sh                 # both suites, all methods; submit and return
#   bash scripts/run_table2.sh ocr             # only one suite: general | ocr (default: both)
#   bash scripts/run_table2.sh tome            # only one method: fixed | random | tome (default: all)
#   bash scripts/run_table2.sh --wait          # block until all tasks finish, then summarize (use inside tmux)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

SUITES=(); WAIT=0; ARRAY=0-35
for arg in "$@"; do
    case "$arg" in
        general|ocr) SUITES+=("$arg") ;;
        fixed)  ARRAY=0-11 ;;
        random) ARRAY=12-23 ;;
        tome)   ARRAY=24-35 ;;
        --wait) WAIT=1 ;;
        *) echo "usage: $0 [general|ocr] [fixed|random|tome] [--wait]"; exit 2 ;;
    esac
done
[[ ${#SUITES[@]} -eq 0 ]] && SUITES=(general ocr)

JIDS=()
for SUITE in "${SUITES[@]}"; do
    # walltime limit per task: general <= 25 min observed; ocr: DocVQA ~2 h at full resolution
    case "$SUITE" in
        general) TIME=1:00:00 ;;
        ocr)     TIME=6:00:00 ;;
    esac
    jid=$(sbatch --parsable --time="$TIME" --job-name="table2_$SUITE" --array="$ARRAY" \
          --export=ALL,SUITE="$SUITE" scripts/eval_table2.sbatch)
    JIDS+=("$jid")
    echo "submitted array $jid (tasks $ARRAY, suite=$SUITE, limit $TIME each). Logs: logs/table2_${SUITE}_${jid}_*.log"
done

if [[ $WAIT -eq 1 ]]; then
    # suites run in parallel: wait for all of them with one no-op job that depends on every array
    deps=$(IFS=:; echo "${JIDS[*]}")
    sbatch --wait --account=PAS2836 --job-name=table2_wait --time=0:01:00 --output=/dev/null \
           --dependency="afterany:$deps" --wrap="true" > /dev/null
    echo "arrays ${JIDS[*]} finished"
    sacct -j "$(IFS=,; echo "${JIDS[*]}")" -X --format=JobID%14,JobName%14,Elapsed,State,ExitCode
    module load miniconda3/24.1.2-py310 && source activate "/fs/scratch/PAS2836/$USER/conda/envs/neo"
    python scripts/summarize_table2.py --latex
else
    echo "Watch: squeue --me"
    echo "When done:  python scripts/summarize_table2.py --latex"
fi
