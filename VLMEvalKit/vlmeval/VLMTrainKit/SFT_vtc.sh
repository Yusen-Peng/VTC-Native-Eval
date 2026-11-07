#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=vtc_sft
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#
# LoRA SFT of NEO-2B with an encoder-level VTC method (fixed | random | drip), the 2x recipe of SFT.sh
# generalised to any ratio. One run = a chain of 4 h segments; each segment resumes from the latest
# checkpoint-* in the output dir (saved every 10 steps), so submit the whole chain at once:
#   j1=$(sbatch --parsable --output=$LOGS/%x_%j.log --export=ALL,VTC_METHOD=drip,RATIO=4 SFT_vtc.sh)
#   j2=$(sbatch --parsable --dependency=afterany:$j1 --output=... --export=... SFT_vtc.sh)   # etc.
# A segment that finds a finished run (trainer_state.json at the output root) exits 0 immediately.
# Smoke (debug-nextgen, a few steps, tests checkpoint save, resume and the final save):
#   sbatch --partition=debug-nextgen --time=0:30:00 \
#     --export=ALL,VTC_METHOD=fixed,RATIO=4,DATASETS=llava_665k%1,SUFFIX=_smoke,MAX_STEPS=2,SAVE_STEPS=2 SFT_vtc.sh
# Env: VTC_METHOD, RATIO (2|4|8), DATASETS (default llava_665k%10), SUFFIX, COMP_LR (compressor lr; unset =
# trainer default, DRIP uses its hardcoded DRIP_LR), MAX_STEPS / SAVE_STEPS (smoke only), REPO (checkout).
set -euo pipefail

VTC_METHOD=${VTC_METHOD:?set VTC_METHOD=fixed|random|drip}
RATIO=${RATIO:?set RATIO=2|4|8}
DATASETS=${DATASETS:-llava_665k%10}
SAVE_STEPS=${SAVE_STEPS:-10}

case "$VTC_METHOD" in
    fixed)  name=Fixed ;;
    random) name=Random ;;
    drip)   name=DRIP ;;
    *) echo "VTC_METHOD must be fixed|random|drip, got $VTC_METHOD"; exit 2 ;;
esac

module load miniconda3/24.1.2-py310
export SCRATCH=/fs/scratch/PAS2836/$USER
source activate "$SCRATCH/conda/envs/neo"

unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE
export HF_HOME=$SCRATCH/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

REPO=${REPO:-$SCRATCH/VTC-Native-Eval}
cd "$REPO/VLMEvalKit/vlmeval/VLMTrainKit"

run_name="NEO_continue_SFT_LLaVA_OV_llava66k_${name}_${RATIO}x${SUFFIX:-}"
output_dir="$SCRATCH/VTC-Native-Eval-ckpts/output/${run_name}"

if [ -z "${MAX_STEPS:-}" ] && [ -f "$output_dir/trainer_state.json" ]; then
    echo "===== $(date) :: ${run_name} already finished ($output_dir/trainer_state.json), nothing to do ====="
    exit 0
fi

NGPU=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | grep -c '^GPU')}
batch_size=1
grad_accum_steps=$((128 / NGPU))
mllm="Paranioar/NEO1_0-2B-SFT"

echo "===== $(date) :: ${run_name} on ${NGPU} GPU(s), grad_accum ${grad_accum_steps}, data ${DATASETS} ====="
echo "commit: $(git -C "$REPO" rev-parse HEAD)  dirty: $(git -C "$REPO" status --porcelain | wc -l)"
echo "resume from: $(ls -d "$output_dir"/checkpoint-* 2>/dev/null | sort -V | tail -1)"

args=(
    --model_name_or_path "${mllm}"
    --tokenizer_name_or_path "${mllm}"
    --dataset_use "${DATASETS}"
    --data_flatten True
    --bf16 True
    --output_dir "${output_dir}"
    --extra_num_layers 12
    --num_hidden_layers 40
    --num_train_epochs 1
    --per_device_train_batch_size "${batch_size}"
    --per_device_eval_batch_size "${batch_size}"
    --gradient_accumulation_steps "${grad_accum_steps}"
    --max_pixels 4194304
    --min_pixels 1310720
    --eval_strategy "no"
    --save_strategy "steps"
    --save_steps "${SAVE_STEPS}"
    --save_total_limit 10
    --learning_rate 2e-5
    --weight_decay 0.0
    --warmup_steps 10
    --max_grad_norm 1
    --logging_steps 1
    --max_seq_length 18432
    --model_max_length 18432
    --gradient_checkpointing True
    --dataloader_num_workers 4
    --run_name "${run_name}"
    --report_to none
    --vtc_method "${VTC_METHOD}"
    --compression_ratio "$(python -c "print(1 / ${RATIO})")"
)
[ "$VTC_METHOD" = drip ] && args+=(--drip_temperature "${DRIP_TEMP:-0.1}")
[ -n "${COMP_LR:-}" ] && args+=(--compressor_learning_rate "${COMP_LR}")
[ -n "${MAX_STEPS:-}" ] && args+=(--max_steps "${MAX_STEPS}")

torchrun \
    --nproc_per_node="${NGPU}" \
    --master_addr=127.0.0.1 \
    --master_port="$(shuf -i 20001-29999 -n 1)" \
    ./neo/train/train.py \
    "${args[@]}"
echo "===== $(date) :: ${run_name} segment finished, checkpoints: $(ls -d "$output_dir"/checkpoint-* 2>/dev/null | xargs -n1 basename | tr '\n' ' ') ====="
