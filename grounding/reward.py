"""Reward composition for SPA-RL: progress contribution + grounding signal.

The final per-step reward used by PPO is

    r_t = w_progress * progress_t + w_grounding * grounding_t + penalty

where ``progress_t`` comes from the Progress Estimator and ``grounding_t``
is the grounding score of the action taken at step ``t``.  Invalid actions
additionally receive an explicit penalty so the policy learns to avoid them.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from grounding.validator import GroundingCheck


@dataclass
class RewardConfig:
    progress_weight: float = 1.0
    grounding_weight: float = 0.5
    invalid_action_penalty: float = -0.2
    success_bonus: float = 0.0


def compose_step_reward(
    progress_value: float,
    grounding_check: Optional[GroundingCheck],
    config: RewardConfig,
) -> float:
    """Combine one progress value with its grounding check into a step reward."""
    reward = config.progress_weight * float(progress_value)
    if grounding_check is not None:
        reward += config.grounding_weight * float(grounding_check.score)
        if not grounding_check.valid:
            reward += config.invalid_action_penalty
    reward += config.success_bonus
    return float(reward)


def grounding_only_rewards(
    checks: List[Optional[GroundingCheck]],
    config: RewardConfig,
) -> List[float]:
    """Grounding part of the reward for a whole trajectory."""
    rewards = []
    for check in checks:
        if check is None:
            rewards.append(0.0)
            continue
        r = config.grounding_weight * float(check.score)
        if not check.valid:
            r += config.invalid_action_penalty
        rewards.append(r)
    return rewards


def compose_trajectory_rewards(
    progress_values: List[float],
    grounding_checks: List[Optional[GroundingCheck]],
    config: RewardConfig,
) -> List[float]:
    """Compose dense rewards for every step of a trajectory."""
    assert len(progress_values) == len(grounding_checks), (
        f"progress ({len(progress_values)}) and grounding ({len(grounding_checks)}) "
        "must have the same length"
    )
    return [
        compose_step_reward(p, g, config)
        for p, g in zip(progress_values, grounding_checks)
    ]
