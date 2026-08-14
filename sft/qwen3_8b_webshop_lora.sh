#!/usr/bin/env bash
# LoRA SFT baseline training for Qwen3-8B on WebShop trajectories.
# This builds the "task basic execution logic" baseline referenced in the
# project plan; the checkpoint is then used as the initial policy for PPO.

set -e

node_num=2                       # number of GPUs
batch_size=16
micro_batch_size=1
accumulation_step=$((batch_size / node_num / micro_batch_size))

export CUDA_VISIBLE_DEVICES=0,1

model_path="Qwen/Qwen3-8B"

deepspeed --include=localhost:0,1 sft/train_sft_lora_qwen3.py \
    --model_name_or_path ${model_path} \
    --data_path data/webshop_sft.json \
    --bf16 True \
    --output_dir ckt/qwen3_8b_webshop_sft \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${micro_batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${accumulation_step} \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess False \
    --deepspeed sft/ds_config_zero3.json \
    --flash_attn False \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj

echo "LoRA SFT finished: ckt/qwen3_8b_webshop_sft"
