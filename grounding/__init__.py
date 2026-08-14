"""SPA-RL Grounding Signal module.

The grounding signal validates every agent action against the current page
state before it is used for reward computation:

  * action parseability        - the action follows the environment grammar;
  * element existence          - the click target / search bar actually exists;
  * button clickability        - the target is not disabled / loading;
  * page loading state         - the page has finished loading;
  * state change               - the action actually changed the observation.

The per-step grounding score is combined with the progress contribution into
the dense reward used by PPO (see ``grounding/reward.py``).
"""

from grounding.validator import GroundingCheck, WebShopGroundingValidator
from grounding.reward import RewardConfig, compose_step_reward, compose_trajectory_rewards

__all__ = [
    "GroundingCheck",
    "WebShopGroundingValidator",
    "RewardConfig",
    "compose_step_reward",
    "compose_trajectory_rewards",
]
