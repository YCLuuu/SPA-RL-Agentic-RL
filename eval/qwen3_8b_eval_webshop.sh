#!/usr/bin/env bash
# Evaluate the Qwen3-8B SPA-RL policy on the WebShop test split.
# Reports: task completion rate, action anchoring accuracy, avg task
# completion time, user intervention rate (see eval_agent/metrics.py).

set -e

MODEL_PATH=${MODEL_PATH:-ckt/qwen3_8b_webshop_merged}
MODEL_NAME=${MODEL_NAME:-qwen3_8b_webshop_merged}
save_path=${SAVE_PATH:-eval/webshop_eval_qwen3/}
logs_path=${save_path}logs
mkdir -p ${logs_path}
task=webshop

# launch the FastChat controller
python -u -m fastchat.serve.controller >> ${logs_path}/model_worker.log 2>&1 &
fs_controller_pid=$!

fs_worker_port=21012
CUDA_VISIBLE_DEVICES=0 python -u -m fastchat.serve.vllm_worker \
    --model-path ${MODEL_PATH} \
    --port ${fs_worker_port} \
    --worker-address http://localhost:${fs_worker_port} >> ${logs_path}/model_worker.log 2>&1 &
fs_worker_pid=$!

sleep 90

python -m eval_agent.main \
    --agent_config fastchat_qwen3 \
    --model_name ${MODEL_NAME} \
    --exp_config ${task} \
    --split test \
    --override \
    --output_path ${save_path}

kill -9 ${fs_worker_pid} 2>/dev/null || true
kill -9 ${fs_controller_pid} 2>/dev/null || true
