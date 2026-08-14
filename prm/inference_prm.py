"""Annotate exploration trajectories with step-wise progress values (PRM inference).

For every assistant turn of every trajectory, the trained Progress Estimator
emits a per-step progress contribution.  The result file is consumed by
``prm/rl_data_org.py`` to build the dense-reward PPO training set.

Example:
    python prm/inference_prm.py \
        --model_path ckt/qwen3_8b_webshop_prm \
        --data_path exploration/webshop/exploration_outputs/exploration.json \
        --output_path prm/exploration_inference_results_webshop.json
"""

import argparse
import json
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from prm.prm_common import SupervisedDataset, prm_model


def calculate_sample_losses(model, dataset, device="cuda:0"):
    """Compute per-step turn values for every trajectory in the dataset."""
    model.to(device)
    model.eval()

    results = []

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Calculating sample loss"):
            sample = dataset[i]

            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
            gpt_unmask = sample["gpt_unmask"].unsqueeze(0).to(device)
            labels = sample["labels"].unsqueeze(0).to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                gpt_unmask=gpt_unmask,
                labels=labels,
            )

            turn_values = outputs.get("turn_values", None)
            if turn_values is not None:
                turn_values = turn_values[0].float().cpu().numpy().tolist()

            final_turn_value = []
            for j in range(len(turn_values)):
                final_turn_value.append(turn_values[j][0])

            raw = dataset.raw_data[i]
            result = {
                "sample_id": i,
                "prediction": outputs["predictions"].item(),
                "ground_truth": labels.item(),
                "loss": outputs["loss"].item(),
                "turn_values": final_turn_value,
                "conversations": raw.get("conversations"),
                "id": raw.get("id"),
                "iteration": raw.get("iteration"),
                "agent_final_reward": raw.get("agent_final_reward"),
                "path": raw.get("path"),
            }
            results.append(result)

    predictions = [r["prediction"] for r in results]
    ground_truths = [r["ground_truth"] for r in results]
    losses = [r["loss"] for r in results]

    avg_loss = np.mean(losses)
    mse = np.mean((np.array(predictions) - np.array(ground_truths)) ** 2)
    mae = np.mean(np.abs(np.array(predictions) - np.array(ground_truths)))
    print(f"\navg_loss={avg_loss:.4f}  mse={mse:.4f}  mae={mae:.4f}")

    return results


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    base_model_path = os.path.join(args.model_path, "our_base_model")
    linear_path = os.path.join(args.model_path, "our_model_state.pt")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    )

    vocab_size = getattr(base_model.config, "vocab_size", None) or len(tokenizer)
    model = prm_model(base_model, vocab_size)

    checkpoint = torch.load(linear_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("Loading dataset...")
    test_data = json.load(open(args.data_path, "r"))
    test_dataset = SupervisedDataset(test_data, tokenizer, model_path=args.model_path)
    test_dataset.raw_data = test_data

    print("Starting step-wise progress annotation...")
    results = calculate_sample_losses(model, test_dataset, device)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Annotate trajectories with step-wise progress values"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="ckt/qwen3_8b_webshop_prm",
        help="Directory containing our_base_model/ and our_model_state.pt",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="exploration/webshop/exploration_outputs/exploration.json",
        help="Exploration trajectories (json) to annotate",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="prm/exploration_inference_results_webshop.json",
        help="Where to write the per-step progress annotations",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    main(parser.parse_args())
