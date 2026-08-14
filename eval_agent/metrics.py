"""Evaluation metrics for SPA-RL (WebShop e-commerce agent).

The module computes the project-level metrics reported in the plan:

  * task completion rate        - fraction of tasks finished successfully;
  * action anchoring accuracy   - fraction of steps whose action was grounded
                                  (parseable, existing element, clickable,
                                  page loaded);
  * avg task completion time    - mean wall-clock seconds per task;
  * user intervention rate      - fraction of tasks where a human correction
                                  was likely needed (>=1 invalid action or the
                                  agent hit the step limit).
"""

from typing import Dict, List


def compute_metrics(states: List) -> Dict[str, float]:
    n = len(states)
    if n == 0:
        return {}

    success = sum(1 for s in states if s.success)
    rewards = [s.reward for s in states if s.reward is not None]
    steps = [s.steps for s in states]
    times = [getattr(s, "total_time", 0.0) for s in states]

    total_actions = 0
    valid_actions = 0
    invalid_tasks = 0
    max_steps_tasks = 0
    element_checks = 0
    element_ok = 0
    click_checks = 0
    click_ok = 0

    for s in states:
        checks = getattr(s, "grounding_checks", [])
        total_actions += len(checks)
        valid_actions += sum(1 for c in checks if c.get("valid"))
        if any(not c.get("valid") for c in checks):
            invalid_tasks += 1
        if s.terminate_reason == "max_steps":
            max_steps_tasks += 1
        for c in checks:
            if c.get("element_exists") is not None:
                element_checks += 1
                if c.get("element_exists"):
                    element_ok += 1
            if c.get("action_type") == "click":
                click_checks += 1
                if c.get("element_exists") is True:
                    click_ok += 1

    return {
        "n_tasks": n,
        "task_completion_rate": success / n,
        "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "avg_steps": sum(steps) / n,
        "avg_task_duration_sec": sum(times) / n,
        "action_anchoring_accuracy": valid_actions / total_actions if total_actions else 0.0,
        "invalid_action_rate": 1.0 - valid_actions / total_actions if total_actions else 0.0,
        "user_intervention_rate": (invalid_tasks + max_steps_tasks) / n,
        "max_steps_rate": max_steps_tasks / n,
        "element_existence_accuracy": element_ok / element_checks if element_checks else 0.0,
        "click_element_accuracy": click_ok / click_checks if click_checks else 0.0,
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    lines = []
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"{k}: {v:.4f}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)
