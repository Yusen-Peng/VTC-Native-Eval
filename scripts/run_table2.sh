#!/bin/bash
# Run on an OSC login node from the repo root on scratch:
#   cd /fs/scratch/PAS2836/$USER/VTC-Native-Eval && git pull --ff-only && bash scripts/run_table2.sh
# Submits one task array per suite, then (optionally) waits for them and prints the Table 2 summary.
#   bash scripts/run_table2.sh                 # 2B, both suites, all compressed models; submit and return
#   bash scripts/run_table2.sh 9B              # scale: 2B | 9B (default: 2B)
#   bash scripts/run_table2.sh ocr             # only one suite: general | ocr (default: both)
#   bash scripts/run_table2.sh tome            # only one method: base | fixed | random | tome | prunesid | spatial
#                                              #   (default: every compressed model; the baseline only with `base`)
#   bash scripts/run_table2.sh --wait          # block until all tasks finish, then summarize (use inside tmux)
# e.g. new scale: bash scripts/run_table2.sh 9B base --wait, check it, then bash scripts/run_table2.sh 9B --wait
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

SCALE=2B; SUITES=(); WAIT=0; ARRAY=4-59
for arg in "$@"; do
    case "$arg" in
        2B|9B)    SCALE=$arg ;;
        general|ocr) SUITES+=("$arg") ;;
        base)     ARRAY=0-3 ;;
        fixed)    ARRAY=4-15 ;;
        random)   ARRAY=16-27 ;;
        tome)     ARRAY=28-39 ;;
        prunesid) ARRAY=40-51 ;;
        spatial)  ARRAY=52-59 ;;
        --wait)   WAIT=1 ;;
        *) echo "usage: $0 [2B|9B] [general|ocr] [base|fixed|random|tome|prunesid|spatial] [--wait]"; exit 2 ;;
    esac
done
[[ ${#SUITES[@]} -eq 0 ]] && SUITES=(general ocr)

JIDS=()
for SUITE in "${SUITES[@]}"; do
    # walltime limit per task: 2B general <= 25 min, ocr DocVQA ~2 h observed; 9B assumed ~2.5-4x slower
    case "$SCALE/$SUITE" in
        2B/general) TIME=1:00:00 ;;
        2B/ocr)     TIME=6:00:00 ;;
        9B/general) TIME=3:00:00 ;;
        9B/ocr)     TIME=12:00:00 ;;
    esac
    jid=$(sbatch --parsable --time="$TIME" --job-name="table2_${SCALE}_$SUITE" --array="$ARRAY" \
          --export=ALL,SCALE="$SCALE",SUITE="$SUITE" scripts/eval_table2.sbatch)
    JIDS+=("$jid")
    echo "submitted array $jid (NEO-$SCALE, tasks $ARRAY, suite=$SUITE, limit $TIME each). Logs: logs/table2_${SCALE}_${SUITE}_${jid}_*.log"
done

if [[ $WAIT -eq 1 ]]; then
    # suites run in parallel: wait for all of them with one no-op job that depends on every array
    deps=$(IFS=:; echo "${JIDS[*]}")
    sbatch --wait --account=PAS2836 --job-name=table2_wait --time=0:01:00 --output=/dev/null \
           --dependency="afterany:$deps" --wrap="true" > /dev/null
    echo "arrays ${JIDS[*]} finished"
    sacct -j "$(IFS=,; echo "${JIDS[*]}")" -X --format=JobID%14,JobName%20,Elapsed,State,ExitCode
    module load miniconda3/24.1.2-py310 && source activate "/fs/scratch/PAS2836/$USER/conda/envs/neo"
    python scripts/summarize_table2.py --scale "$SCALE" --latex
else
    echo "Watch: squeue --me"
    echo "When done:  python scripts/summarize_table2.py --scale $SCALE --latex"
fi