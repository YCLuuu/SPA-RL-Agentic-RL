import re
import json
import logging
from typing import Tuple

from eval_agent.envs import BaseEnv
from eval_agent.tasks import WebShopTask
from eval_agent.prompt import prompt_with_icl
from eval_agent.utils.datatypes import State
from grounding.validator import WebShopGroundingValidator
from webshop.web_agent_site.envs import WebAgentTextEnv


logger = logging.getLogger("agent_frame")


class WebShopEnv(BaseEnv):
    def __init__(
        self,
        task: WebShopTask,
        env: WebAgentTextEnv,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task: WebShopTask = task
        self.session_id = self.task.session_id
        self.session = {}
        self.env = env
        self.grounding_validator = WebShopGroundingValidator()
        
        self.state = State()
    
    def parse_action(self, llm_output: str) -> str:
        llm_output = llm_output.strip()
        pattern = re.compile(r"Action: (.*)", re.DOTALL)
        action = re.findall(pattern, llm_output)[0]
        assert action is not None
        return action
    
    def step(self, llm_output: str) -> Tuple[str, State]:
        self.state.start_step()
        self.state.history.append({
            "role": "assistant",
            "content": llm_output
        })
        try:
            action = self.parse_action(llm_output)
        except:
            observation = f"Observation: Invalid format. The input must contains 'Action: '"
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
            self.state.steps += 1
            self.state.reward = 0
            if self.state.steps >= self.max_steps:
                self.state.finished = True
                self.state.success = False
                self.state.terminate_reason = "max_steps"
                self.state.reward = 0
            self.state.end_step()
            return observation, self.state
        # ---- Grounding signal: validate the action against the page state ----
        try:
            page_info = self.env.get_available_actions()
        except Exception:
            page_info = {}
        clickables = page_info.get("clickables")
        has_search_bar = page_info.get("has_search_bar")
        prev_observation = None
        for msg in reversed(self.state.history[:-1]):
            if msg.get("role") == "user":
                prev_observation = msg.get("content")
                break
        grounding_check = self.grounding_validator.check(
            action,
            clickables=clickables,
            has_search_bar=has_search_bar,
            observation=self.env.observation if hasattr(self.env, "observation") else None,
            prev_observation=prev_observation,
        )
        self.state.grounding_checks.append(grounding_check.to_dict())
        if not grounding_check.valid:
            self.state.invalid_action_count += 1
        try:
            observation, reward, done, info = self.env.step(action=action)
            observation = f"Observation:\n{observation}"
            # available_actions = self.env.get_available_actions()
            # observation = f"Observation:\n{observation}\n\nAvailable Actions:\n{available_actions}"
        except AssertionError:
            observation = 'Observation: Invalid action!'
            reward = 0
            done = False

        self.state.history.append({
            "role": "user",
            "content": f"{observation}",
        })

        self.state.steps += 1
        if self.state.steps >= self.max_steps:
            self.state.finished = True
            self.state.success = False
            self.state.terminate_reason = "max_steps"
            self.state.reward = 0

        if done:
            self.state.finished = True
            self.state.success = True
            self.state.terminate_reason = "success"
            self.state.reward = reward
        else:
            self.state.reward = reward

        self.state.end_step()
        return observation, self.state
    
    def reset(self) -> Tuple[str, State]:
        self.state = State()
        self.env.reset(self.session_id)
        cur_task = self.env.observation
        observation, messages = prompt_with_icl(self.instruction, self.raw_icl, cur_task, 1)
        if self.icl_format == 'first':
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
        elif self.icl_format == 'conversation':
            self.state.history = messages
        return observation, self.state
