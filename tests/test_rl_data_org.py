import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grounding.reward import RewardConfig
from prm.rl_data_org import build_rl_samples


def test_build_rl_samples():
    annotations = json.load(open("data/demo/inference_results_demo.json"))
    sampled = build_rl_samples(
        annotations,
        template_name="qwen3",
        reward_config=RewardConfig(
            progress_weight=1.0, grounding_weight=0.5, invalid_action_penalty=-0.2
        ),
        max_turns=35,
        max_samples=100,
    )
    assert len(sampled["prompt"]) == 3
    assert len(sampled["response"]) == 3
    assert len(sampled["reward"]) == 3
    for prompts, responses, rewards in zip(
        sampled["prompt"], sampled["response"], sampled["reward"]
    ):
        assert len(prompts) == len(responses) == len(rewards)
        # Qwen3 ChatML template
        assert prompts[0].startswith("<|im_start|>user")
        assert responses[0].startswith("<|im_start|>assistant")


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
