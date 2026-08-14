import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grounding.reward import RewardConfig, compose_step_reward, compose_trajectory_rewards
from grounding.validator import WebShopGroundingValidator


def test_compose_step_reward_valid():
    config = RewardConfig(progress_weight=1.0, grounding_weight=0.5)
    v = WebShopGroundingValidator()
    check = v.check("click[Buy Now]", clickables=["buy now"])
    r = compose_step_reward(0.3, check, config)
    assert r > 0.3  # grounding adds positive signal


def test_compose_step_reward_invalid_penalty():
    config = RewardConfig(
        progress_weight=1.0, grounding_weight=0.5, invalid_action_penalty=-0.2
    )
    v = WebShopGroundingValidator()
    check = v.check("click[NOPE]", clickables=["buy now"])
    r = compose_step_reward(0.3, check, config)
    # 0.3*1.0 + 0.5*score(<=0.35) - 0.2  < 0.3
    assert r < 0.3


def test_trajectory_rewards():
    config = RewardConfig()
    v = WebShopGroundingValidator()
    checks = [
        v.check("search[perfume]", has_search_bar=True),
        v.check("click[B000VOHH8I]", clickables=["b000vohh8i"]),
        v.check("click[Buy Now]", clickables=["buy now"]),
    ]
    rewards = compose_trajectory_rewards([0.1, 0.4, 0.5], checks, config)
    assert len(rewards) == 3
    assert all(r > 0 for r in rewards)


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
