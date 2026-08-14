"""Verify the Qwen3 (ChatML) label-masking algorithm without transformers.

The real ``preprocess`` in ``prm/prm_common.py`` / ``fastchat/train/train.py``
needs a HuggingFace tokenizer; here we emulate tokenization with a fake
tokenizer and re-run the same masking loop to assert that only assistant
response tokens remain as labels.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IGNORE_TOKEN_ID = -100


class FakeTokenizer:
    """Minimal tokenizer: splits special tokens, then whitespace."""

    def __init__(self, vocab, pad_token_id=0, model_max_length=64):
        self.vocab = vocab
        self.pad_token_id = pad_token_id
        self.model_max_length = model_max_length

    def _tokenize(self, text):
        specials = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
        parts = [text]
        for sp in specials:
            new_parts = []
            for p in parts:
                new_parts.extend(re.split(f"({re.escape(sp)})", p))
            parts = new_parts
        toks = []
        for p in parts:
            if p in specials:
                toks.append(p)
            elif p.strip():
                toks.extend(p.split())
        return toks

    def __call__(self, texts, return_tensors=None, padding=None, max_length=None, truncation=None):
        max_len = max_length or self.model_max_length
        rows = []
        for text in texts:
            toks = self._tokenize(text)
            seq = [self.vocab[t] for t in toks][:max_len]
            seq += [self.pad_token_id] * (max_len - len(seq))
            rows.append(seq)
        return {"input_ids": rows}


def build_vocab():
    vocab = {}

    def add(tok):
        if tok not in vocab:
            vocab[tok] = len(vocab)

    add("<|pad|>")
    for sp in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
        add(sp)
    for word in (
        "system",
        "user",
        "assistant",
        "You",
        "are",
        "a",
        "helpful",
        "assistant.",
        "find",
        "a",
        "product",
        "Thought:",
        "search",
        "for",
        "it",
        "Action:",
        "search[foo]",
        "click[Buy",
        "Now]",
        "more",
        "results",
        "click",
        "it",
    ):
        add(word)
    return vocab


def apply_chatml_mask(conversation, tokenizer):
    input_ids = tokenizer([conversation], padding="max_length", truncation=True)["input_ids"][0]
    targets = list(input_ids)
    sep2 = "<|im_end|>"

    turns = conversation.split(sep2)
    cur_len = 0
    for turn in turns:
        if turn == "":
            break
        turn = turn + sep2
        turn_len = len(tokenizer._tokenize(turn))
        if "<|im_start|>system" in turn or "<|im_start|>user" in turn:
            targets[cur_len: cur_len + turn_len] = [IGNORE_TOKEN_ID] * turn_len
        cur_len += turn_len
    targets[cur_len:] = [IGNORE_TOKEN_ID] * (len(targets) - cur_len)
    return targets, input_ids


def test_chatml_mask_keeps_only_assistant():
    vocab = build_vocab()
    tokenizer = FakeTokenizer(vocab, pad_token_id=vocab["<|pad|>"], model_max_length=64)

    conversation = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nfind a product<|im_end|>\n"
        "<|im_start|>assistant\nThought: search for it\nAction: search[foo]<|im_end|>\n"
        "<|im_start|>user\nmore results<|im_end|>\n"
        "<|im_start|>assistant\nThought: click it\nAction: click[Buy Now]<|im_end|>\n"
    )

    targets, input_ids = apply_chatml_mask(conversation, tokenizer)
    non_ignored = [i for i, t in enumerate(targets) if t != IGNORE_TOKEN_ID]

    # Everything after the last assistant turn is padding -> masked.
    assert len(non_ignored) > 0
    # Decode the unmasked region: it must be exactly the two assistant turns.
    unmasked_text = ""
    for i in non_ignored:
        tok_id = input_ids[i]
        unmasked_text += [k for k, v in vocab.items() if v == tok_id][0] + " "
    compact = unmasked_text.replace(" ", "")
    assert "Action:search[foo]" in compact
    assert "Action:click[BuyNow]" in compact
    assert "findaproduct" not in compact
    assert "Youareahelpfulassistant." not in compact


if __name__ == "__main__":
    import traceback

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    sys.exit(1 if failures else 0)
