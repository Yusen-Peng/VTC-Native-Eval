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
#   sbatch --export=ALL,RATIO=2,QUERIES=learned_residual SFT_perceiver.sh   # pooled queries + learned latent grid (init 0)
#   sbatch --export=ALL,RATIO=2,QUERIES=learned,INIT_FROM=$SCRATCH/VTC-Native-Eval-ckpts/init/perceiver_learned_warm_2x.bin SFT_perceiver.sh
#       # learned queries warm-started from scripts/perceiver_warmstart.py (run name PerceiverLearnedWarm, COMP_LR 1e-3)
#   sbatch --export=ALL,RATIO=4,COMP_LR=1e-3 SFT_perceiver.sh      # pooled queries, compressor lr 1e-3 (run name PerceiverLr1e3)
# RATIO = 2 | 4 | 8. COMP_LR = compressor learning rate (default: 2e-5 for pooled, 1e-4 for learned, 1e-3 for
# learned_residual and for learned + INIT_FROM; LoRA stays 2e-5).
# Smoke test (~15 min, no full run):
#   sbatch --partition=debug-nextgen --time=0:15:00 --export=ALL,RATIO=2,DATASETS=llava_665k%1,SUFFIX=_smoke SFT_perceiver.sh
# ~1.45 min/step on 1 A100 -> 488 steps ~ 11.8 h. Global batch = NGPU x grad_accum = 128 regardless of
# --gpus-per-node, but >1 GPU (DDP) is untested in this training code.
# Re-submitting the same RATIO resumes from the latest checkpoint-* in the output dir, so a full run can be a
# chain of 4 h segments (`sbatch --time=4:00:00 ... ; sbatch --dependency=afterany:<prev> ...`); a segment that
# finds the final perceiver.bin (written only after the last step) exits 0 without loading anything.
# REPO = checkout to train from (default: the shared $SCRATCH/VTC-Native-Eval); its VLMEvalKit is put first on
# PYTHONPATH so `import vlmeval` resolves there and not to the editable install of another checkout.
set -euo pipefail

RATIO=${RATIO:?set RATIO=2|4|8}
DATASETS=${DATASETS:-llava_665k%10}

module load miniconda3/24.1.2-py310
export SCRATCH=/fs/scratch/PAS2836/$USER
source activate "$SCRATCH/conda/envs/neo"

unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE
export HF_HOME=$SCRATCH/hf
export HF_HUB_OFFLINE=1

REPO=${REPO:-$SCRATCH/VTC-Native-Eval}
export PYTHONPATH="$REPO/VLMEvalKit:$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO/VLMEvalKit/vlmeval/VLMTrainKit"

# Count allocated GPUs from Slurm; `nvidia-smi -L | wc -l` overcounts on MIG nodes (extra partition lines)
NGPU=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | grep -c '^GPU')}
batch_size=1
grad_accum_steps=$((128 / NGPU))
mllm="Paranioar/NEO1_0-2B-SFT"
QUERIES=${QUERIES:-pooled}
USER_COMP_LR=${COMP_LR:-}
case "$QUERIES" in
    pooled)  name=Perceiver;        COMP_LR=${COMP_LR:-2e-5}
             # pooled queries at a non-default compressor lr get their own run name (PerceiverLr1e3 for 1e-3), so the
             # original Perceiver_{2,4,8}x runs are never resumed or overwritten; config key ...-PerceiverLr1e3-<r>x
             [ -n "$USER_COMP_LR" ] && [ "$USER_COMP_LR" != 2e-5 ] && name="PerceiverLr$(echo "$USER_COMP_LR" | tr -d '.-')" ;;
    learned) name=PerceiverLearned; COMP_LR=${COMP_LR:-1e-4} ;;
    learned_residual) name=PerceiverLearnedResidual; COMP_LR=${COMP_LR:-1e-3} ;;
    *) echo "QUERIES must be pooled|learned|learned_residual, got $QUERIES"; exit 2 ;;
esac
INIT_FROM=${INIT_FROM:-}
if [ -n "$INIT_FROM" ]; then
    [ "$QUERIES" = learned ] || { echo "INIT_FROM is only wired for QUERIES=learned"; exit 2; }
    [ -f "$INIT_FROM" ] || { echo "INIT_FROM not found: $INIT_FROM"; exit 2; }
    name=PerceiverLearnedWarm
    COMP_LR=${USER_COMP_LR:-1e-3}
fi
run_name="NEO_continue_SFT_LLaVA_OV_llava66k_${name}_${RATIO}x${SUFFIX:-}"
output_dir="$SCRATCH/VTC-Native-Eval-ckpts/output/${run_name}"

echo "===== $(date) :: ${run_name} on ${NGPU} GPU(s), grad_accum ${grad_accum_steps}, data ${DATASETS}, compressor lr ${COMP_LR}, init ${INIT_FROM:-fresh} ====="
echo "commit: $(git rev-parse HEAD)  repo: $REPO"
if [ -f "$output_dir/perceiver.bin" ] && [ -f "$output_dir/adapter_model.safetensors" ]; then
    echo "training already complete ($output_dir has the final perceiver.bin + adapter), nothing to do"
    exit 0
fi
echo "checkpoints present: $(ls -d "$output_dir"/checkpoint-* 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"

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
[ -n "$INIT_FROM" ] && args+=(--perceiver_init_from "$INIT_FROM")

torchrun \
    --nproc_per_node="${NGPU}" \
    --master_addr=127.0.0.1 \
    --master_port="$(shuf -i 20001-29999 -n 1)" \
    ./neo/train/train.py \
    "${args[@]}"
