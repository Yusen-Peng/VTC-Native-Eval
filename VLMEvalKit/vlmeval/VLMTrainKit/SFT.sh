#!/bin/bash
#SBATCH --job-name=NEO_continue_SFT_LLaVA_OV_CLEVR_fixed_2x
#SBATCH --output=NEO_continue_SFT_LLaVA_OV_CLEVR_fixed_2x.log
#SBATCH --account=PAS2836
#SBATCH --partition=debug-nextgen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00

module load miniconda3/24.1.2-py310
conda deactivate
conda activate neo

cd /users/PAS2912/yusenpeng/VTC-Eval/VLMEvalKit/vlmeval/VLMTrainKit

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
mllm="Paranioar/NEO1_0-2B-SFT"

lr=2e-5
batch_size=1
grad_accum_steps=128

entry_file="./neo/train/train.py"

datasets="clevr_llava_onevision%1"

run_name="NEO_continue_SFT_LLaVA_OV_CLEVR_fixed_2x"
output_dir="/fs/scratch/PAS2836/yusenpeng_checkpoint/VTC-Native-Eval-ckpts/output/${run_name}"

args=(
    --model_name_or_path "${mllm}"
    --tokenizer_name_or_path "${mllm}"
    --dataset_use "${datasets}"
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
    --learning_rate "${lr}"
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
    --vtc_method fixed
    --compression_ratio 0.5
)

torchrun \
    --nproc_per_node=1 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${entry_file}" \
    "${args[@]}"