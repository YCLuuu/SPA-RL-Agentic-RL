"""LoRA supervised fine-tuning for Qwen3-8B (WebShop / other agentic tasks).

This is a light-weight adaptation of ``fastchat/train/train_lora.py`` for
Qwen3 models: it attaches LoRA adapters to all attention + MLP projections,
uses the Qwen3 (ChatML) conversation template for label masking, and saves a
standalone PEFT adapter that can be merged back with ``ppo/merge.py``.

Example (4 GPUs, DeepSpeed ZeRO-3):
    deepspeed --include=localhost:0,1,2,3 sft/train_sft_lora_qwen3.py \
        --model_name_or_path Qwen/Qwen3-8B \
        --data_path data/webshop_sft.json \
        --bf16 True \
        --output_dir ckt/qwen3_8b_webshop_sft \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --model_max_length 4096 \
        --gradient_checkpointing True \
        --deepspeed sft/ds_config_zero3.json \
        --lora_r 16 --lora_alpha 32 \
        --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
"""

import logging
import os
import pathlib
import sys
import typing
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import transformers
from transformers import HfArgumentParser, Trainer
import torch
from peft import LoraConfig, get_peft_model

from fastchat.train.train import (
    DataArguments,
    ModelArguments,
    make_supervised_data_module,
)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: typing.Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512)
    flash_attn: bool = field(default=False)


@dataclass
class LoraArguments:
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: typing.List[str] = field(default=None)
    lora_weight_path: str = field(default="")
    lora_bias: str = field(default="none")


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    if lora_args.lora_target_modules is None:
        # Default: cover all linear projections used by Qwen3 blocks.
        lora_args.lora_target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation="flash_attention_2" if training_args.flash_attn else None,
    )

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        model_path=model_args.model_name_or_path,
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    model.config.use_cache = False
    trainer.train()
    trainer.save_state()

    if training_args.local_rank in (0, -1):
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        print(f"LoRA adapter saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
