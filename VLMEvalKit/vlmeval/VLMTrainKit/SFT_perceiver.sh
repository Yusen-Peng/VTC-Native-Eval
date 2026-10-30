#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=perceiver_sft
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --output=logs/%x_%j.log
#
# LoRA SFT of NEO-2B with a trained Perceiver resampler (compressors/perceiver.py), same recipe as SFT.sh (DRIP).
# Submit from VLMEvalKit/vlmeval/VLMTrainKit (mkdir -p logs first), one job per compression ratio:
#   sbatch --export=ALL,RATIO=2 SFT_perceiver.sh                   # pooled queries (start = fixed pooling)
#   sbatch --export=ALL,RATIO=2,QUERIES=learned SFT_perceiver.sh   # learned latent queries (as in the paper)
# RATIO = 2 | 4 | 8. COMP_LR = compressor learning rate (default: 2e-5 for pooled, 1e-4 for learned; LoRA stays 2e-5).
# Smoke test (~15 min, no full run):
#   sbatch --partition=debug-nextgen --time=0:15:00 --export=ALL,RATIO=2,DATASETS=llava_665k%1,SUFFIX=_smoke SFT_perceiver.sh
# ~1.4 min/step on 1 A100 -> 488 steps ~ 11.5 h. Global batch = NGPU x grad_accum = 128 regardless of
# --gpus-per-node, but >1 GPU (DDP) is untested in this training code.
# Re-submitting the same RATIO resumes from the latest checkpoint-* in the output dir.
set -euo pipefail

RATIO=${RATIO:?set RATIO=2|4|8}
DATASETS=${DATASETS:-llava_665k%10}

module load miniconda3/24.1.2-py310
export SCRATCH=/fs/scratch/PAS2836/$USER
source activate "$SCRATCH/conda/envs/neo"

unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE
export HF_HOME=$SCRATCH/hf
export HF_HUB_OFFLINE=1

cd "$SCRATCH/VTC-Native-Eval/VLMEvalKit/vlmeval/VLMTrainKit"

NGPU=$(nvidia-smi -L | wc -l)
batch_size=1
grad_accum_steps=$((128 / NGPU))
mllm="Paranioar/NEO1_0-2B-SFT"
QUERIES=${QUERIES:-pooled}
case "$QUERIES" in
    pooled)  name=Perceiver;        COMP_LR=${COMP_LR:-2e-5} ;;
    learned) name=PerceiverLearned; COMP_LR=${COMP_LR:-1e-4} ;;
    *) echo "QUERIES must be pooled|learned, got $QUERIES"; exit 2 ;;
esac
run_name="NEO_continue_SFT_LLaVA_OV_llava66k_${name}_${RATIO}x${SUFFIX:-}"
output_dir="$SCRATCH/VTC-Native-Eval-ckpts/output/${run_name}"

echo "===== $(date) :: ${run_name} on ${NGPU} GPU(s), grad_accum ${grad_accum_steps}, data ${DATASETS}, compressor lr ${COMP_LR} ====="
echo "commit: $(git rev-parse HEAD)"

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
    --save_steps 10
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
    --vtc_method perceiver
    --perceiver_queries "${QUERIES}"
    --compressor_learning_rate "${COMP_LR}"
    --compression_ratio "$(python -c "print(1 / ${RATIO})")"
)

torchrun \
    --nproc_per_node="${NGPU}" \
    --master_addr=127.0.0.1 \
    --master_port="$(shuf -i 20001-29999 -n 1)" \
    ./neo/train/train.py \
    "${args[@]}"
