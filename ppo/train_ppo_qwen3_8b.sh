#!/usr/bin/env bash
# PPO training for Qwen3-8B with SPA dense rewards
# (progress contribution + grounding signal).

set -e

export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES="0,1"

export MODEL_TYPE="qwen3-8b"
export MODEL_PATH="ckt/qwen3_8b_webshop_sft_merged"

torchrun \
    --nproc_per_node 2 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6603 \
    ppo/step_ppo.py \
    --model_path ${MODEL_PATH} \
    --model_type ${MODEL_TYPE} \
    --config_path config/StepTool_ppo_qwen3_8b.json \
    --data_file prm/sampled_data_rl_training_webshop_qwen3.json \
    --epochs 1 \
    --max_context_len 4096 \
    --max_response_len 1200
