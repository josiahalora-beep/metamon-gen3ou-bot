"""Instrumented, Generation 3-aware tactical support for Metamon.

The package is deliberately independent of AMAGO. It consumes poke-env battle
objects at the integration boundary and returns an action index to the existing
Metamon action space.
"""

from .state import BattleState, snapshot_battle
from .damage import DamageRange, calculate_damage
from .speed import speed_range, can_outspeed, speed_tie_probability
from .opponent_model import OpponentModel, PredictedResponse
from .response_search import ResponseSearcher, ResponseScore
from .strict_evaluator import TacticalEvaluator
from .logger import BattleLogger

__all__ = [
    "BattleState", "snapshot_battle", "DamageRange", "calculate_damage",
    "speed_range", "can_outspeed", "speed_tie_probability",
    "OpponentModel", "PredictedResponse", "ResponseSearcher", "ResponseScore",
    "TacticalEvaluator", "BattleLogger",
]
