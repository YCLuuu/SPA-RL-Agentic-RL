import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_agent.metrics import compute_metrics
from eval_agent.utils.datatypes import State


def _make_state(success, steps, invalid=0, total_time=10.0):
    s = State(reward=1.0 if success else 0.0, finished=True, success=success)
    s.steps = steps
    s.total_time = total_time
    s.step_times = [total_time / steps] * steps if steps else []
    for _ in range(steps):
        s.grounding_checks.append(
            {"valid": True, "element_exists": True, "action_type": "click"}
        )
    for idx in range(1, invalid + 1):
        s.grounding_checks[-idx]["valid"] = False
        s.grounding_checks[-idx]["element_exists"] = False
    s.invalid_action_count = invalid
    return s


def test_compute_metrics():
    states = [
        _make_state(success=True, steps=4, invalid=0, total_time=20.0),
        _make_state(success=True, steps=5, invalid=1, total_time=30.0),
        _make_state(success=False, steps=10, invalid=2, total_time=60.0),
    ]
    m = compute_metrics(states)
    assert m["n_tasks"] == 3
    assert abs(m["task_completion_rate"] - 2 / 3) < 1e-9
    assert abs(m["avg_task_duration_sec"] - (20 + 30 + 60) / 3) < 1e-9
    total_actions = 4 + 5 + 10
    valid_actions = 4 + 4 + 8
    assert abs(m["action_anchoring_accuracy"] - valid_actions / total_actions) < 1e-9
    assert m["user_intervention_rate"] == 2 / 3  # two tasks need intervention


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
