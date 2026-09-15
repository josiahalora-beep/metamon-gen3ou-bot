from __future__ import annotations

from typing import Any

from . import evaluator as evaluator_module
from .evaluator import TacticalEvaluator as _BaseTacticalEvaluator
from .strategic import SafetyDecision


class TacticalEvaluator(_BaseTacticalEvaluator):
    """Production evaluator with the learned policy kept dominant.

    Hard mechanical loss checks remain active. Broad heuristic safety is not
    allowed to re-enter after predictive search, because recorded ladder data
    showed that repeated heuristic overrides were the main source of negative
    counterfactual deltas.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        search = self.response_search
        # Raise the bar for predictive overrides. These are intentionally
        # instance attributes so benchmarks can still construct the base class
        # unchanged when testing individual layers.
        search._KNOWN_RESPONSE_MIN = 0.12
        search._STRONG_SWITCH_PROB = 0.70
        search._SPECIFIC_SWITCH_PROB = 0.25
        search._MOVE_OVERRIDE_DAMAGE_GAIN = 30.0
        search._MOVE_OVERRIDE_KO_GAIN = 0.35
        search._SWITCH_OVERRIDE_MARGIN = 8.0

    def evaluate(self, battle: Any, legal_actions: list[int], model_action: int):
        original_safety = evaluator_module.safety_override
        evaluator_module.safety_override = lambda *args, **kwargs: SafetyDecision(None)
        try:
            return super().evaluate(battle, legal_actions, model_action)
        finally:
            evaluator_module.safety_override = original_safety
