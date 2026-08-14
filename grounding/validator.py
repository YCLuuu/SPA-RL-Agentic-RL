"""Grounding Signal: per-step action validity checking for WebShop-style envs.

The validator is deliberately environment-agnostic in its interface: the
caller supplies the parsed action plus whatever page facts are available
(clickable elements, disabled buttons, loading state, before/after
observations).  When used online (eval env) the WebShop environment provides
authoritative ``clickables``; when used offline (trajectory annotation) the
clickables are inferred heuristically from the stored observation text.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set


ACTION_PATTERN = re.compile(r"^(search|click)\[(.*)\]$", re.DOTALL)

# Known WebShop controls that always appear as clickables when present.
KNOWN_CONTROLS = {
    "buy now",
    "next >",
    "< prev",
    "back to search",
    "search",
    "add to cart",
    "description",
    "features",
    "reviews",
    "size",
    "color",
}

LOADING_INDICATORS = (
    "loading",
    "please wait",
    "spinner",
    "page not found",
    "error 404",
)


def default_weights() -> Dict[str, float]:
    return {
        "parseable": 0.20,
        "element_exists": 0.30,
        "button_clickable": 0.20,
        "page_loaded": 0.15,
        "state_changed": 0.15,
    }


@dataclass
class GroundingCheck:
    """Result of validating one agent action against the page state."""

    action: str
    action_type: Optional[str] = None
    argument: Optional[str] = None
    parseable: bool = False
    element_exists: Optional[bool] = None
    button_clickable: Optional[bool] = None
    page_loaded: bool = True
    state_changed: Optional[bool] = None
    valid: bool = False
    score: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict:
        return {
            "action": self.action,
            "action_type": self.action_type,
            "argument": self.argument,
            "parseable": self.parseable,
            "element_exists": self.element_exists,
            "button_clickable": self.button_clickable,
            "page_loaded": self.page_loaded,
            "state_changed": self.state_changed,
            "valid": self.valid,
            "score": float(self.score),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "GroundingCheck":
        return cls(
            action=d.get("action", ""),
            action_type=d.get("action_type"),
            argument=d.get("argument"),
            parseable=d.get("parseable", False),
            element_exists=d.get("element_exists"),
            button_clickable=d.get("button_clickable"),
            page_loaded=d.get("page_loaded", True),
            state_changed=d.get("state_changed"),
            valid=d.get("valid", False),
            score=d.get("score", 0.0),
            reason=d.get("reason", ""),
        )


def extract_clickables_from_observation(observation: Optional[str]) -> Set[str]:
    """Heuristically infer clickable labels from a WebShop text observation.

    In "simple" text mode every visible segment is separated by ``[SEP]``;
    buttons/links/options are among those segments.  We keep segments that
    look like controls, product ids, or option labels and normalize them.
    """
    if not observation:
        return set()
    segments = [s.strip() for s in observation.split("[SEP]") if s.strip()]
    clickables: Set[str] = set()
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        norm = seg.lower()
        # Product option ids look like "b0xxxx..." (asins), control names are
        # short, and product titles are long — keep the short candidates and
        # every known control.
        if (
            norm in KNOWN_CONTROLS
            or len(seg) <= 40
            or re.fullmatch(r"[a-z0-9]{8,12}", norm)
        ):
            clickables.add(norm)
    return clickables


def normalize_argument(arg: Optional[str]) -> str:
    return "" if arg is None else arg.strip().lower()


class WebShopGroundingValidator:
    """Validates WebShop actions and produces a grounding score in [0, 1]."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = default_weights() if weights is None else weights

    def parse_action(self, action: str):
        m = ACTION_PATTERN.match(action.strip())
        if m is None:
            return None, None
        return m.group(1).lower(), m.group(2).strip()

    def _element_exists(
        self,
        action_type: str,
        argument: str,
        clickables: Optional[Iterable[str]],
        has_search_bar: Optional[bool],
    ) -> Optional[bool]:
        if action_type == "search":
            if has_search_bar is None:
                return None
            return bool(has_search_bar)
        if clickables is None:
            return None
        clickable_set = {normalize_argument(c) for c in clickables}
        return argument in clickable_set or argument.strip("[]") in clickable_set

    def check(
        self,
        action: str,
        clickables: Optional[Iterable[str]] = None,
        has_search_bar: Optional[bool] = None,
        disabled: Optional[Iterable[str]] = None,
        observation: Optional[str] = None,
        prev_observation: Optional[str] = None,
        page_loading: Optional[bool] = None,
    ) -> GroundingCheck:
        action_type, argument = self.parse_action(action)
        parseable = action_type is not None and bool(argument)

        check = GroundingCheck(
            action=action,
            action_type=action_type,
            argument=argument,
            parseable=parseable,
        )

        if not parseable:
            check.reason = "unparseable action"
            check.valid = False
            check.score = 0.0
            return check

        # 1. Element existence -------------------------------------------------
        if clickables is None and observation is not None:
            clickables = extract_clickables_from_observation(observation)
        check.element_exists = self._element_exists(
            action_type, normalize_argument(argument), clickables, has_search_bar
        )

        # 2. Button clickability ------------------------------------------------
        if action_type == "click" and disabled is not None:
            disabled_set = {normalize_argument(d) for d in disabled}
            check.button_clickable = normalize_argument(argument) not in disabled_set
        elif action_type == "search":
            check.button_clickable = True

        # 3. Page loading state -------------------------------------------------
        if page_loading is not None:
            check.page_loaded = not page_loading
        elif observation is not None:
            obs_low = observation.lower()
            check.page_loaded = not any(ind in obs_low for ind in LOADING_INDICATORS)

        # 4. State change -------------------------------------------------------
        if prev_observation is not None:
            check.state_changed = (observation or "") != (prev_observation or "")

        # 5. Composite validity and score --------------------------------------
        check.valid = (
            check.element_exists is not False
            and check.button_clickable is not False
            and check.page_loaded
        )

        dims = {
            "parseable": 1.0,
            "element_exists": (
                1.0 if check.element_exists else 0.0 if check.element_exists is False else None
            ),
            "button_clickable": (
                1.0
                if check.button_clickable
                else 0.0
                if check.button_clickable is False
                else None
            ),
            "page_loaded": 1.0 if check.page_loaded else 0.0,
            "state_changed": (
                1.0 if check.state_changed else 0.0 if check.state_changed is False else None
            ),
        }
        w_sum = 0.0
        score = 0.0
        for name, w in self.weights.items():
            v = dims.get(name)
            if v is None:
                continue
            score += w * v
            w_sum += w
        if w_sum > 0:
            score /= w_sum

        # Hard caps: an action that violates grounding cannot keep a high score.
        if check.element_exists is False:
            score = min(score, 0.35)
            check.reason = "element does not exist on the current page"
        elif check.button_clickable is False:
            score = min(score, 0.55)
            check.reason = "button is disabled / not clickable"
        elif not check.page_loaded:
            score = min(score, 0.55)
            check.reason = "page is still loading"
        else:
            check.reason = "ok"

        check.score = float(score)
        return check
