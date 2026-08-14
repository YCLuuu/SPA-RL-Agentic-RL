"""Shared building blocks for the SPA-RL Progress Estimator (PRM).

The progress estimator is a base LLM with a scalar head over the last token
of every assistant turn.  It is trained to predict the *sum* of per-step
progress values, i.e. the terminal (delayed) reward of a trajectory.  At
inference time, the per-step values become the dense intermediate rewards
used by PPO.

This module is shared by:
  - prm/train_our_progress_model.py  (train the estimator)
  - prm/inference_prm.py             (annotate step-wise progress)

Both Qwen3 (ChatML) and Llama-3 conversation formats are supported.
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import transformers
from torch.utils.data import Dataset
from transformers import PreTrainedModel
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template, get_model_adapter


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class prm_model(PreTrainedModel):
    """Base LLM backbone + a linear scalar head over the LM logits.

    The head is applied to the logits of the last token of each assistant
    turn; the per-turn values are summed to predict the trajectory-level
    (delayed) reward, and the per-turn values themselves are the step-wise
    progress contributions.
    """

    def __init__(self, base_model, vocab_size=None):
        super().__init__(base_model.config)
        self.backbone = base_model
        vocab_size = vocab_size or getattr(base_model.config, "vocab_size", None)
        if vocab_size is None:
            raise ValueError("vocab_size must be provided when the model config has no vocab_size")
        self.LN = nn.Linear(vocab_size, 1).to(torch.bfloat16)
        self.config.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask, gpt_unmask=None, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask).logits

        batch_predictions = []
        batch_turn_values = []

        for i in range(outputs.size(0)):
            sample_labels = gpt_unmask[i]
            valid_indices = torch.where(sample_labels != -100)[0]

            if len(valid_indices) == 0:
                batch_predictions.append(torch.zeros(1, device=outputs.device, dtype=outputs.dtype))
                batch_turn_values.append(torch.zeros(0, device=outputs.device, dtype=outputs.dtype))
                continue

            # Find the last token of every assistant turn: a jump in the
            # valid-index sequence marks the boundary between two turns.
            turn_end_indices = []
            for j in range(1, len(valid_indices)):
                if valid_indices[j] - valid_indices[j - 1] > 1:
                    turn_end_indices.append(valid_indices[j - 1])
            turn_end_indices.append(valid_indices[-1])

            turn_logits = torch.stack([outputs[i, idx, :] for idx in turn_end_indices])
            turn_values = self.LN(turn_logits)
            sample_prediction = turn_values.sum()

            batch_predictions.append(sample_prediction.unsqueeze(0))
            batch_turn_values.append(turn_values)

        value_outputs = torch.cat(batch_predictions)

        loss = None
        if labels is not None:
            loss = nn.MSELoss()(value_outputs, labels)

        return {
            "loss": loss,
            "predictions": value_outputs,
            "turn_values": batch_turn_values,
        }


def read_json(source):
    print(f"Reading file: {source}")
    json_list = []
    with open(source, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
        f.seek(0)
        for line in f:
            json_list.append(json.loads(line))
    return json_list


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    model_path: str,
) -> Dict:
    """Tokenize conversations and produce ``gpt_unmask`` masks.

    Positions that belong to assistant responses keep their token id; every
    other position (system prompt, user turns, padding) is set to
    ``IGNORE_TOKEN_ID``.  The PRM forward pass uses these masks to locate the
    end of each assistant turn.
    """
    conv = get_model_adapter(model_path).get_default_conv_template(model_path)
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # --- Qwen1/Qwen2/Qwen3 (ChatML) -------------------------------------
    if conv.sep_style == SeparatorStyle.CHATML:
        sep2 = "<|im_end|>"

        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            turns = conversation.split(sep2)
            cur_len = 0
            for turn in turns:
                if turn == "":
                    break
                turn = turn + sep2
                turn_len = len(tokenizer(turn).input_ids)
                if "<|im_start|>system" in turn or "<|im_start|>user" in turn:
                    target[cur_len: cur_len + turn_len] = IGNORE_TOKEN_ID
                cur_len += turn_len
            target[cur_len:] = IGNORE_TOKEN_ID

            if cur_len < tokenizer.model_max_length and cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" #turn = {len(turns) - 1}. (ignored)"
                )

        return dict(
            input_ids=input_ids,
            gpt_unmask=targets,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )

    # --- Llama 3 ---------------------------------------------------------
    if conv.sep_style == SeparatorStyle.LLAMA3:
        sep2 = "<|eot_id|>"
        sep = "<|end_header_id|>"

        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            turns = conversation.split(sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_TOKEN_ID
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                if i % 2 == 0:
                    instruction_len = len(tokenizer(turn).input_ids)
                    target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
                    cur_len += instruction_len
                else:
                    parts = turn.split(sep)
                    turn_len = len(tokenizer(turn).input_ids)
                    if len(parts) != 2:
                        break
                    instruction_len = len(tokenizer(parts[0]).input_ids)
                    target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
                    cur_len += turn_len
            target[cur_len:] = IGNORE_TOKEN_ID

            if cur_len < tokenizer.model_max_length and cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" #turn = {len(turns) - 1}. (ignored)"
                )

        return dict(
            input_ids=input_ids,
            gpt_unmask=targets,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )

    raise NotImplementedError(
        f"Conversation template {conv.name} is not supported by the PRM pipeline"
    )


class SupervisedDataset(Dataset):
    """Dataset of trajectories with trajectory-level (delayed) reward labels."""

    def __init__(self, raw_data, tokenizer, model_path: Optional[str] = None):
        super(SupervisedDataset, self).__init__()
        print("Formatting inputs...")
        self.raw_data = raw_data
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, model_path)

        self.input_ids = data_dict["input_ids"]
        self.gpt_unmask = data_dict["gpt_unmask"]
        self.attention_mask = data_dict["attention_mask"]

        self.labels = torch.tensor(
            [each_piece["agent_final_reward"] for each_piece in raw_data],
            dtype=torch.bfloat16,
        )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            gpt_unmask=self.gpt_unmask[i],
            attention_mask=self.attention_mask[i],
            labels=self.labels[i],
        )
