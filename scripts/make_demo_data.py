"""Generate synthetic WebShop demo data for the SPA-RL pipeline.

Produces:
  * data/demo/webshop_sft_demo.json       - SFT conversation data
  * data/demo/exploration_demo.json       - exploration trajectories (PRM)
  * data/demo/inference_results_demo.json - fake PRM step annotations

This lets the whole data pipeline (grounding annotation -> reward
composition -> PPO data format) be dry-run without GPUs or the real WebShop
data download.
"""

import json
import os


SUCCESS_TRAJECTORY = [
    {
        "from": "human",
        "value": (
            "Observation:\nWebShop [SEP] Instruction: [SEP] i need a long lasting "
            "6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars "
            "[SEP] Search"
        ),
    },
    {
        "from": "gpt",
        "value": "Thought: I should search for the product.\nAction: search[l'eau d'issey 6.76 fl oz]",
    },
    {
        "from": "human",
        "value": (
            "Observation:\nInstruction: [SEP] i need a long lasting 6.76 fl oz bottle "
            "of l'eau d'issey, and price lower than 100.00 dollars [SEP] Back to Search "
            "[SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B000VOHH8I [SEP] "
            "L'eau D'issey By Issey Miyake for Men Eau De Toilette Spray [SEP] $64.98 "
            "[SEP] B000MJZOPK [SEP] L'eau d'Issey by Issey Miyake for Women 3.3 oz [SEP] $49.98"
        ),
    },
    {
        "from": "gpt",
        "value": "Thought: B000VOHH8I matches the goal.\nAction: click[B000VOHH8I]",
    },
    {
        "from": "human",
        "value": (
            "Observation:\nInstruction: [SEP] i need a long lasting 6.76 fl oz bottle "
            "of l'eau d'issey, and price lower than 100.00 dollars [SEP] Back to Search "
            "[SEP] < Prev [SEP] size [SEP] 2.5 fl oz [SEP] 6.76 fl oz (pack of 1) [SEP] "
            "L'eau D'issey By Issey Miyake for Men Eau De Toilette Spray [SEP] Price: $64.98 "
            "[SEP] Rating: N.A. [SEP] Buy Now"
        ),
    },
    {
        "from": "gpt",
        "value": "Thought: Select the 6.76 fl oz option.\nAction: click[6.76 fl oz (pack of 1)]",
    },
    {
        "from": "human",
        "value": (
            "Observation:\nInstruction: [SEP] i need a long lasting 6.76 fl oz bottle "
            "of l'eau d'issey, and price lower than 100.00 dollars [SEP] Back to Search "
            "[SEP] < Prev [SEP] size [SEP] 2.5 fl oz [SEP] 6.76 fl oz (pack of 1) [SEP] "
            "L'eau D'issey By Issey Miyake for Men Eau De Toilette Spray [SEP] Price: $64.98 "
            "[SEP] Rating: N.A. [SEP] Buy Now"
        ),
    },
    {
        "from": "gpt",
        "value": "Thought: The option matches the requirement, buy it.\nAction: click[Buy Now]",
    },
]


INVALID_ACTION_TRAJECTORY = [
    SUCCESS_TRAJECTORY[0],
    SUCCESS_TRAJECTORY[1],
    SUCCESS_TRAJECTORY[2],
    {
        "from": "gpt",
        "value": "Thought: Click the product.\nAction: click[Z999NONEXISTENT]",
    },
    {
        "from": "human",
        "value": "Observation: Invalid action!",
    },
    {
        "from": "gpt",
        "value": "Thought: I should retry.\nAction: click[B000VOHH8I]",
    },
    SUCCESS_TRAJECTORY[4],
    SUCCESS_TRAJECTORY[5],
    SUCCESS_TRAJECTORY[6],
    SUCCESS_TRAJECTORY[7],
]


UNPARSEABLE_TRAJECTORY = [
    SUCCESS_TRAJECTORY[0],
    {
        "from": "gpt",
        "value": "Thought: Let me try something.\nAction: press the red button",
    },
    {
        "from": "human",
        "value": "Observation: Invalid format. The input must contains 'Action: '",
    },
    SUCCESS_TRAJECTORY[1],
    SUCCESS_TRAJECTORY[2],
    SUCCESS_TRAJECTORY[3],
    {
        "from": "human",
        "value": (
            "Observation:\nInstruction: [SEP] i need a long lasting 6.76 fl oz bottle "
            "of l'eau d'issey, and price lower than 100.00 dollars [SEP] Back to Search "
            "[SEP] < Prev [SEP] size [SEP] 2.5 fl oz [SEP] 6.76 fl oz (pack of 1) [SEP] "
            "L'eau D'issey By Issey Miyake for Men Eau De Toilette Spray [SEP] Price: $64.98 "
            "[SEP] Rating: N.A. [SEP] Buy Now"
        ),
    },
    {
        "from": "gpt",
        "value": "Thought: Select the size.\nAction: click[6.76 fl oz (pack of 1)]",
    },
    {
        "from": "human",
        "value": (
            "Observation:\nInstruction: [SEP] i need a long lasting 6.76 fl oz bottle "
            "of l'eau d'issey, and price lower than 100.00 dollars [SEP] Back to Search "
            "[SEP] < Prev [SEP] size [SEP] 2.5 fl oz [SEP] 6.76 fl oz (pack of 1) [SEP] "
            "L'eau D'issey By Issey Miyake for Men Eau De Toilette Spray [SEP] Price: $64.98 "
            "[SEP] Rating: N.A. [SEP] Buy Now"
        ),
    },
    {
        "from": "gpt",
        "value": "Thought: Buy it.\nAction: click[Buy Now]",
    },
]


def split_reward(num_steps: int, final_reward: float) -> list:
    """Distribute a terminal reward into fake per-step progress values."""
    if num_steps == 0:
        return []
    # Simulate a progress estimator: last step gets the largest share.
    weights = [0.5 + 0.5 * (i + 1) / num_steps for i in range(num_steps)]
    total = sum(weights)
    return [final_reward * w / total for w in weights]


def main():
    os.makedirs("data/demo", exist_ok=True)
    os.makedirs("prm", exist_ok=True)

    trajectories = [
        {
            "id": "demo-0001",
            "iteration": 0,
            "agent_final_reward": 1.0,
            "success": True,
            "path": "demo/0001",
            "conversations": SUCCESS_TRAJECTORY,
        },
        {
            "id": "demo-0002",
            "iteration": 0,
            "agent_final_reward": 0.5,
            "success": True,
            "path": "demo/0002",
            "conversations": INVALID_ACTION_TRAJECTORY,
        },
        {
            "id": "demo-0003",
            "iteration": 0,
            "agent_final_reward": 0.0,
            "success": False,
            "path": "demo/0003",
            "conversations": UNPARSEABLE_TRAJECTORY,
        },
    ]

    with open("data/demo/exploration_demo.json", "w", encoding="utf-8") as f:
        json.dump(trajectories, f, ensure_ascii=False, indent=2)

    sft_data = [{"conversations": t["conversations"]} for t in trajectories]
    with open("data/demo/webshop_sft_demo.json", "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    inference_results = []
    for t in trajectories:
        n_turns = len(t["conversations"]) // 2
        turn_values = split_reward(n_turns, t["agent_final_reward"])
        inference_results.append(
            {
                "sample_id": len(inference_results),
                "prediction": sum(turn_values),
                "ground_truth": t["agent_final_reward"],
                "loss": 0.0,
                "turn_values": turn_values,
                "conversations": t["conversations"],
                "id": t["id"],
                "iteration": t["iteration"],
                "agent_final_reward": t["agent_final_reward"],
                "path": t["path"],
            }
        )

    with open("data/demo/inference_results_demo.json", "w", encoding="utf-8") as f:
        json.dump(inference_results, f, ensure_ascii=False, indent=2)

    print("Demo data written to data/demo/")


if __name__ == "__main__":
    main()
