"""Train the SPA-RL Progress Estimator (PRM).

The estimator learns to decompose the terminal (delayed) reward of an agent
trajectory into per-step progress contributions.  It is trained with an MSE
loss between the *sum* of predicted per-step values and the trajectory-level
final reward.

Example (Qwen3-8B, 4 GPUs):
    python prm/data_org.py   # organize exploration trajectories first
    deepspeed --include=localhost:0,1,2,3 prm/train_our_progress_model.py \
        --model_path ckt/qwen3_8b_webshop_sft \
        --data_path exploration/webshop/exploration_outputs/exploration.json \
        --output_dir ckt/qwen3_8b_webshop_prm
"""

import argparse
import json
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from prm.prm_common import SupervisedDataset, prm_model


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mse = np.mean((predictions - labels) ** 2)
    mae = np.mean(np.abs(predictions - labels))
    accuracy = np.mean(np.abs(predictions - labels) <= 0.1)
    return {"mse": float(mse), "mae": float(mae), "accuracy": float(accuracy)}


def main(args):
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    print("Loading data...")
    train_data = json.load(open(args.train_path, "r"))
    val_data = json.load(open(args.val_path, "r"))
    test_data = json.load(open(args.val_path, "r"))

    train_dataset = SupervisedDataset(train_data, tokenizer, model_path=args.model_path)
    val_dataset = SupervisedDataset(val_data, tokenizer, model_path=args.model_path)
    test_dataset = SupervisedDataset(test_data, tokenizer, model_path=args.model_path)

    print("Creating model instance...")
    vocab_size = getattr(base_model.config, "vocab_size", None) or len(tokenizer)
    model = prm_model(base_model, vocab_size)

    deepspeed_config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": "auto",
        "optimizer": {
            "type": "Adam",
            "params": {"lr": 3e-6, "weight_decay": 0.01},
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 3e-6,
                "warmup_num_steps": 500,
            },
        },
        "gradient_clipping": 1.0,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
            # PRM trains the full backbone: Adam states (2x fp32 copies of
            # 8B params) do not fit in 40GB, so keep them on CPU.  The 16.4GB
            # fp16/bf16 weights themselves fit on each A100-40GB.
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
        },
        "steps_per_print": 10,
        "wall_clock_breakdown": False,
        "fp16": {"enabled": False},
        "amp": {"enabled": False},
    }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="no",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=args.learning_rate,
        bf16=True,
        save_safetensors=False,
        deepspeed=deepspeed_config,
        gradient_accumulation_steps=args.grad_accum,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    model_save_path = args.output_dir
    os.makedirs(model_save_path, exist_ok=True)

    base_model_save_path = os.path.join(model_save_path, "our_base_model")
    base_model.save_pretrained(base_model_save_path)
    tokenizer.save_pretrained(base_model_save_path)

    model_state_dict = {
        "model_state_dict": model.state_dict(),
        "config": model.config,
    }
    torch.save(model_state_dict, os.path.join(model_save_path, "our_model_state.pt"))
    print(f"Model saved to: {model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SPA-RL Progress Estimator")
    parser.add_argument(
        "--model_path",
        type=str,
        default="ckt/qwen3_8b_webshop_sft_merged",
        help="SFT model path (Qwen3-8B LoRA merged checkpoint)",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="exploration/webshop/exploration_outputs/exploration.json",
        help="Path to organized exploration trajectories for training",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="exploration/webshop/exploration_outputs/exploration_tiny.json",
        help="Path to validation trajectories",
    )
    parser.add_argument("--output_dir", type=str, default="ckt/qwen3_8b_webshop_prm")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=3e-6)
    main(parser.parse_args())
