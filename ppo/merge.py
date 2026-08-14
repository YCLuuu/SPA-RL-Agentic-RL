"""Merge a LoRA adapter back into the base LLM to obtain the final model.

Works with any decoder-only base model (Llama-3, Qwen3, ...).  Used both for
merging the SFT adapter into the baseline and for merging the PPO value-head
checkpoint into the RL policy.

Example:
    python ppo/merge.py \
        --base_model_path ckt/qwen3_8b_webshop_sft \
        --adapter_path ckt/qwen3_8b_webshop_prm/checkpoint \
        --output_dir ckt/qwen3_8b_webshop_merged
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model = model.merge_and_unload()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model merged and saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA into the base LLM")
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    main(parser.parse_args())
