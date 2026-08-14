"""Build the dense-reward PPO training set for SPA-RL.

For every exploration trajectory the script combines:

  * step-wise progress contributions from the Progress Estimator
    (``prm/exploration_inference_results_webshop.json``), and
  * grounding scores from the Grounding Signal module (annotated on the fly
    or read from ``--grounding_annotations``),

into the weighted dense reward used by PPO:

    r_t = w_progress * progress_t + w_grounding * grounding_t + penalty

Example:
    python prm/rl_data_org.py \
        --inference_results prm/exploration_inference_results_webshop.json \
        --template qwen3 \
        --output prm/sampled_data_rl_training_webshop_qwen3.json
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grounding.annotate import annotate_trajectory, compute_grounding_metrics
from grounding.reward import RewardConfig, compose_step_reward
from grounding.validator import GroundingCheck, WebShopGroundingValidator


TEMPLATES = {
    "qwen3": {
        "human": "<|im_start|>user\n{system_message}<|im_end|>\n",
        "gpt": "<|im_start|>assistant\n{system_message}<|im_end|>\n",
    },
    "llama3": {
        "human": "<|start_header_id|>user<|end_header_id|>\n\n{system_message}<|eot_id|>",
        "gpt": "{system_message}<|eot_id|>",
    },
}


def build_rl_samples(
    annotations: List[Dict],
    template_name: str = "qwen3",
    reward_config: RewardConfig = None,
    max_turns: int = 35,
    max_samples: int = 1596,
) -> Dict:
    reward_config = reward_config or RewardConfig()
    templates = TEMPLATES[template_name]
    validator = WebShopGroundingValidator()

    all_prompts: List[List[str]] = []
    all_responses: List[List[str]] = []
    all_rewards: List[List[float]] = []

    for annotation in annotations:
        conversations = annotation["conversations"]
        turn_values = annotation.get("turn_values", [])
        if len(conversations) % 2 != 0:
            continue
        if len(conversations) // 2 != len(turn_values):
            continue

        # Grounding checks: one per assistant turn.
        if "grounding_checks" not in annotation:
            annotation = annotate_trajectory(annotation, validator)
        grounding_checks = annotation.get("grounding_checks", [])
        if len(grounding_checks) != len(turn_values):
            continue
        grounding_checks = [
            g if isinstance(g, GroundingCheck) else GroundingCheck.from_dict(g)
            for g in grounding_checks
        ]

        prompts, responses, rewards = [], [], []
        for j in range(len(turn_values)):
            prompts.append(
                templates["human"].format(system_message=conversations[2 * j]["value"])
            )
            responses.append(
                templates["gpt"].format(system_message=conversations[2 * j + 1]["value"])
            )
            reward = compose_step_reward(
                turn_values[j],
                grounding_checks[j],
                reward_config,
            )
            rewards.append(reward)

        all_prompts.append(prompts)
        all_responses.append(responses)
        all_rewards.append(rewards)

    # Sampling: cap both the number of trajectories and the trajectory length.
    keep = [i for i in range(len(all_rewards)) if len(all_rewards[i]) < max_turns]

    random.seed(42)
    if len(keep) > max_samples:
        keep = random.sample(keep, max_samples)

    sampled = {
        "prompt": [all_prompts[i] for i in keep],
        "response": [all_responses[i] for i in keep],
        "reward": [all_rewards[i] for i in keep],
    }
    return sampled


def main(args):
    with open(args.inference_results, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    print(f"Loaded {len(annotations)} PRM annotations")

    reward_config = RewardConfig(
        progress_weight=args.progress_weight,
        grounding_weight=args.grounding_weight,
        invalid_action_penalty=args.invalid_action_penalty,
    )

    # Annotate grounding up-front so the grounding metrics can be reported.
    validator = WebShopGroundingValidator()
    annotated = []
    for annotation in annotations:
        if "grounding_checks" not in annotation:
            annotation = annotate_trajectory(annotation, validator)
        annotated.append(annotation)

    metrics = compute_grounding_metrics(annotated)
    print(
        "Grounding metrics: "
        + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
    )

    sampled = build_rl_samples(
        annotated,
        template_name=args.template,
        reward_config=reward_config,
        max_turns=args.max_turns,
        max_samples=args.max_samples,
    )
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(sampled['prompt'])} trajectories -> {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SPA-RL dense-reward PPO data")
    parser.add_argument(
        "--inference_results",
        type=str,
        default="prm/exploration_inference_results_webshop.json",
        help="PRM step-wise progress annotations",
    )
    parser.add_argument("--output", type=str, default="prm/sampled_data_rl_training_webshop_qwen3.json")
    parser.add_argument("--template", type=str, choices=["qwen3", "llama3"], default="qwen3")
    parser.add_argument("--progress_weight", type=float, default=1.0)
    parser.add_argument("--grounding_weight", type=float, default=0.5)
    parser.add_argument("--invalid_action_penalty", type=float, default=-0.2)
    parser.add_argument("--max_turns", type=int, default=35)
    parser.add_argument("--max_samples", type=int, default=1596)
    args = parser.parse_args()
    main(args)
