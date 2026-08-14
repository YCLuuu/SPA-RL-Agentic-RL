# SPA-RL（Qwen3-8B / WebShop）服务器实验手册

目标硬件：**Linux 服务器，4× A100-40GB**。代码已按此配置（SFT/PPO 4 卡、
DeepSpeed 去 offload、PRM 优化器 offload + 微批 1）。

按顺序执行，每步都给出验证点。所有路径默认在仓库根目录 `spa-rl/` 下。

---

## 0. 服务器验收

```bash
nvidia-smi                 # 应看到 4 张 A100 (40GB)，驱动 ≥ 525
df -h /                    # 预留 ≥ 200GB 磁盘（模型+数据+ckpt）
free -g                    # 内存建议 ≥ 128GB（PRM 全参训练优化器在 CPU）
```

## 1. 代码与基础模型下载

```bash
# 代码（已有则跳过）
git clone <你的仓库地址> spa-rl && cd spa-rl

# Qwen3-8B（国内推荐 ModelScope，或 hf-mirror）
pip install -U modelscope
modelscope download --model Qwen/Qwen3-8B --local_dir ./models/Qwen3-8B

# 若用 HuggingFace：
# export HF_ENDPOINT=https://hf-mirror.com
# huggingface-cli download Qwen/Qwen3-8B --local-dir ./models/Qwen3-8B
```

然后把 `sft/qwen3_8b_webshop_lora.sh` 里的模型路径改为本地路径：

```bash
sed -i 's|model_path="Qwen/Qwen3-8B"|model_path="./models/Qwen3-8B"|' sft/qwen3_8b_webshop_lora.sh
```

## 2. 创建两套 conda 环境

```bash
conda create -n SPA python=3.9 -y && conda activate SPA
pip install -r requirements_autodl.txt   # AutoDL/A100 专用（不含 flash_attn）
# webshop 环境：faiss-cpu==1.7.4 在镜像源上已不可用且运行时不需它，用 --no-deps
cd envs/webshop && pip install -e . --no-deps && cd ../..
pip install beautifulsoup4==4.11.1 cleantext==1.1.4 env==0.1.0 \
    Flask==2.1.2 gym==0.24.0 rank_bm25==0.2.2 \
    spacy==3.6.1 thinc==8.1.12 thefuzz==0.20.0 werkzeug==2.0.3 \
    selenium==4.2.0 requests_mock pytest numpy==1.26.3   # 与 webshop/setup.py 对齐
pip install "pyserini==0.17.0"           # 会自带宽泛的 faiss-cpu，Lucene 检索不需要 faiss
python -m spacy download en_core_web_lg
conda install -y -c conda-forge openjdk=11

conda create -n RL_train python=3.10 -y && conda activate RL_train
pip install -r ppo/requirements.txt
```

> 本管线已**不依赖 flash_attn**（源码编译在 pip 隔离环境里必失败），
> SFT/PRM/PPO 全部走 PyTorch 原生 attention，A100 上运行无影响。

验证：`python -c "import torch, transformers; print(torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0))"`
应输出 `2.5.x 4 NVIDIA A100-PCIE-40GB` 之类（torch 必须是 2.5.x，不能是 2.6+）。

> vLLM 装完若把 torch 顶到 2.6+，用下面命令装回：
> `pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121`

## 3. WebShop 数据下载与放置

```bash
cd envs/webshop
gdown https://drive.google.com/uc?id=1G_0ccLWn5kZE5rpeyAdh_YuoNzvBUjT9   # data.zip
gdown https://drive.google.com/uc?id=11zOUDkJSgGhYin9NxQtG8PVpDsika86y  # indexes.zip
unzip data.zip
mkdir -p search_index
unzip indexes.zip -d search_index/
cd ../..

# 专家轨迹（SFT 数据 + 探索轨迹参考）
gdown https://drive.google.com/uc?id=1_tBMDixZcIjKuv-LExNllha-YIRxhKIq
unzip data.zip
```

> gdown 需要能访问 Google Drive；国内服务器挂代理或找镜像源。
> 若 `unzip data.zip` 后 `data/` 目录已存在，回答覆盖即可（两个 zip 都叫
> data.zip，解压目标不同：第一个在 `envs/webshop/`，第二个在仓库根）。

### 数据布局校验（重要）

```bash
# WebShop 环境数据
ls envs/webshop/data/items_shuffle.json envs/webshop/data/items_ins_v2.json \
   envs/webshop/data/reviews.json envs/webshop/search_index/ | head

# SFT 数据（两处都要有，缺失就复制一份）
ls data/webshop_sft.json || echo "缺 data/webshop_sft.json"
ls sft/data/webshop_sft.json || cp data/webshop_sft.json sft/data/webshop_sft.json
```

若 `envs/webshop` 下解压后数据不在 `envs/webshop/data/`，而代码期望
`envs/webshop/src/data/`，建软链：

```bash
ln -s ../data envs/webshop/src/data   # 按实际位置调整
```

## 4. 数据管线冒烟（可选，10 秒）

```bash
python scripts/make_demo_data.py
python tests/run_all.py                # 17 个测试全过即可
```

## 5. SFT baseline（Qwen3-8B + LoRA）

```bash
conda activate SPA
bash sft/qwen3_8b_webshop_lora.sh
# 产出：ckt/qwen3_8b_webshop_sft/（LoRA adapter）
```

## 6. 合并 SFT 权重（baseline 完整模型）

```bash
python ppo/merge.py \
  --base_model_path ./models/Qwen3-8B \
  --adapter_path ckt/qwen3_8b_webshop_sft \
  --output_dir ckt/qwen3_8b_webshop_sft_merged
```

## 7. 基线评估（建立 baseline 指标）

```bash
MODEL_PATH=ckt/qwen3_8b_webshop_sft_merged \
MODEL_NAME=qwen3_8b_webshop_sft_merged \
SAVE_PATH=eval/webshop_eval_qwen3_baseline \
bash eval/qwen3_8b_eval_webshop.sh

cat eval/webshop_eval_qwen3_baseline/metrics.json
# 记录 task_completion_rate 等，这是方案里的性能 baseline
```

## 8. 探索轨迹采集（SFT 模型探索 WebShop）

```bash
bash exploration/webshop/my_generate_response_webshop.sh
# 产出：exploration/webshop/exploration_outputs/explore/*.json
```

脚本已按 4 卡配置：4 个 vLLM worker（GPU 0-3）+ 16 个探索进程，约数小时。

## 9. 轨迹整理（PRM 训练数据）

```bash
python prm/data_org.py
# 产出：exploration/webshop/exploration_outputs/exploration.json（含 tiny 版）
```

## 10. 训练 Progress Estimator

```bash
deepspeed --include=localhost:0,1,2,3 prm/train_our_progress_model.py \
  --model_path ckt/qwen3_8b_webshop_sft_merged \
  --train_path exploration/webshop/exploration_outputs/exploration.json \
  --val_path exploration/webshop/exploration_outputs/exploration_tiny.json \
  --output_dir ckt/qwen3_8b_webshop_prm
# 产出：ckt/qwen3_8b_webshop_prm/（our_base_model/ + our_model_state.pt）
```

## 11. 步级进度预测（稠密奖励来源）

```bash
python prm/inference_prm.py \
  --model_path ckt/qwen3_8b_webshop_prm \
  --data_path exploration/webshop/exploration_outputs/exploration.json \
  --output_path prm/exploration_inference_results_webshop.json
```

## 12. 构造 PPO 稠密奖励数据（进度 + Grounding）

```bash
python prm/rl_data_org.py \
  --inference_results prm/exploration_inference_results_webshop.json \
  --template qwen3 \
  --progress_weight 1.0 \
  --grounding_weight 0.5 \
  --invalid_action_penalty -0.2 \
  --output prm/sampled_data_rl_training_webshop_qwen3.json
# 会打印 grounding 指标（动作锚定准确率等）
```

## 13. PPO 训练

```bash
conda activate RL_train
export PYTHONPATH=./
bash ppo/train_ppo_qwen3_8b.sh
# 产出：ckpts/steptool_qwen3-8b_*/ 下的 LoRA checkpoint
```

> 默认开 wandb；无 wandb 账号时给命令加 `--no_wandb`。

## 14. 合并 PPO 权重并最终评估

```bash
# 找到最新的 PPO checkpoint
ls ckpts/steptool_qwen3-8b_*/

conda activate SPA
python ppo/merge.py \
  --base_model_path ckt/qwen3_8b_webshop_sft_merged \
  --adapter_path <上面找到的 checkpoint 目录> \
  --output_dir ckt/qwen3_8b_webshop_merged

bash eval/qwen3_8b_eval_webshop.sh
cat eval/webshop_eval_qwen3/metrics.json
```

对比第 7 步的 baseline 与这里的最终 metrics.json，即方案要求的：
任务完成率、动作锚定准确率、平均完成时长、用户干预率。

---

## 常见问题

| 现象 | 处理 |
| --- | --- |
| gdown 下载失败 | 服务器需要能访问 Google Drive（挂代理）；或找国内镜像 |
| `flash_attn` 相关报错 | 本管线已不依赖 flash_attn；确认 requirements_autodl.txt 无 flash_attn、SFT 脚本 `--flash_attn False`、PRM 不指定 flash_attention_2 |
| 端口冲突 | controller(21001)/worker(21012)/PPO(6603) 被占时改脚本端口 |
| 显存 OOM | SFT/PPO 已按 40GB 调过；若仍 OOM 把 `--max_context_len 4096` 降到 2048 |
| wandb 登录卡住 | 在命令后加 `--no_wandb`，并把 `log_with` 改 "tensorboard" |
| vLLM worker 起不来 | 看 `eval/webshop_eval_qwen3/logs/model_worker.log`，多为模型路径或显存 |
| 探索阶段模型名不匹配 | 确认 `ckt/qwen3_8b_webshop_sft_merged` 存在且与脚本 `cur_model_name` 一致 |

## 预计耗时（4× A100-40GB）

| 阶段 | 预估 |
| --- | --- |
| SFT 3 epoch（LoRA） | 1-3 小时 |
| 探索轨迹采集 | 2-6 小时（视任务数） |
| PRM 训练 1 epoch | 2-4 小时 |
| PPO 1 epoch | 1-3 小时 |
| 评估 | 30-60 分钟 |
