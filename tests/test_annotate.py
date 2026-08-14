import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grounding.annotate import annotate_trajectory, compute_grounding_metrics, extract_action


def test_extract_action():
    assert extract_action("Thought: hi\nAction: click[Buy Now]") == "click[Buy Now]"
    assert extract_action("Action: search[foo]") == "search[foo]"


def test_annotate_demo_trajectory():
    data = json.load(open("data/demo/exploration_demo.json"))
    annotated = [annotate_trajectory(t) for t in data]
    for item in annotated:
        assert "grounding_checks" in item
        n_turns = len(item["conversations"]) // 2
        assert len(item["grounding_checks"]) == n_turns

    metrics = compute_grounding_metrics(annotated)
    assert metrics["n_actions"] > 0
    assert 0.0 <= metrics["action_anchoring_accuracy"] <= 1.0
    assert metrics["invalid_action_rate"] > 0  # demo contains invalid actions


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
