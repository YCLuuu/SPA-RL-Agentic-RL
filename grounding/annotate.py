"""Offline grounding annotation for stored exploration trajectories.

Every assistant turn of an exploration trajectory is validated against the
observation that preceded it (and, when available, the observation that
followed it).  The resulting ``grounding_checks`` are stored back into the
trajectory file and consumed by ``prm/rl_data_org.py`` to build the dense
PPO reward signal.
"""

import json
import re
from typing import Dict, List, Optional

from grounding.validator import GroundingCheck, WebShopGroundingValidator


ACTION_RE = re.compile(r"Action:\s*(.+?)\s*$", re.DOTALL)


def extract_action(llm_output: str) -> str:
    """Extract the ``Action: ...`` line from an LLM response."""
    m = ACTION_RE.search(llm_output or "")
    if m is None:
        return (llm_output or "").strip()
    return m.group(1).strip()


def annotate_trajectory(
    item: Dict,
    validator: Optional[WebShopGroundingValidator] = None,
) -> Dict:
    """Add a ``grounding_checks`` field to one exploration trajectory."""
    validator = validator or WebShopGroundingValidator()
    conversations = item.get("conversations", [])
    checks: List[Dict] = []

    for j in range(0, len(conversations) - 1, 2):
        if conversations[j].get("from") != "human":
            continue
        obs_before = conversations[j].get("value", "")
        gpt_msg = conversations[j + 1].get("value", "")
        if conversations[j + 1].get("from") != "gpt":
            continue

        action = extract_action(gpt_msg)
        obs_after = None
        if j + 2 < len(conversations):
            obs_after = conversations[j + 2].get("value", "")

        check: GroundingCheck = validator.check(
            action,
            clickables=item.get("available_actions") if "available_actions" in item else None,
            observation=obs_before,
            prev_observation=obs_after,
        )
        checks.append(check.to_dict())

    item = dict(item)
    item["grounding_checks"] = checks
    return item


def annotate_file(
    input_path: str,
    output_path: str,
    validator: Optional[WebShopGroundingValidator] = None,
) -> List[Dict]:
    """Annotate a JSON file containing a list of exploration trajectories."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotated = [annotate_trajectory(item, validator) for item in data]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotated, f, ensure_ascii=False, indent=2)
    print(f"Annotated {len(annotated)} trajectories -> {output_path}")
    return annotated


def compute_grounding_metrics(
    items: List[Dict],
) -> Dict[str, float]:
    """Aggregate grounding statistics over annotated trajectories.

    ``action_anchoring_accuracy`` is the fraction of actions that were valid
    (parseable, targeting an existing element, clickable, page loaded).  It is
    the offline proxy for the project metric "动作锚定准确率".
    """
    total = 0
    valid = 0
    parseable = 0
    exists = 0
    exists_total = 0
    changed = 0
    changed_total = 0

    for item in items:
        for check in item.get("grounding_checks", []):
            total += 1
            if check.get("valid"):
                valid += 1
            if check.get("parseable"):
                parseable += 1
            if check.get("element_exists") is True:
                exists += 1
            if check.get("element_exists") is not None:
                exists_total += 1
            if check.get("state_changed") is True:
                changed += 1
            if check.get("state_changed") is not None:
                changed_total += 1

    return {
        "n_actions": total,
        "action_anchoring_accuracy": valid / total if total else 0.0,
        "parseable_rate": parseable / total if total else 0.0,
        "element_existence_accuracy": exists / exists_total if exists_total else 0.0,
        "state_change_rate": changed / changed_total if changed_total else 0.0,
        "invalid_action_rate": 1.0 - valid / total if total else 0.0,
    }
