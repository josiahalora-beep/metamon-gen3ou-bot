from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .damage import calculate_damage
from .opponent_model import OpponentModel, PredictedResponse
from metamon.interface import consistent_move_order, consistent_pokemon_order


@dataclass(frozen=True)
class ResponseScore:
    action: int
    score: float
    opponent_expected: float
    rationale: str

    @property
    def expected(self) -> float:
        return self.opponent_expected


class ResponseSearcher:
    """Bounded response search anchored to the learned policy.

    The learned action is the anchor. Prediction may only override it when
    the current action is mechanically sound and the predicted response is
    sufficiently specific and valuable. Action indices always use the same
    canonical ordering as the main evaluator.
    """

    _MOVE_TYPES = {
        "surf": "water", "hydropump": "water", "icebeam": "ice", "thunderbolt": "electric",
        "thunder": "electric", "psychic": "psychic", "fireblast": "fire", "flamethrower": "fire",
        "gigadrain": "grass", "leafstorm": "grass", "energyball": "grass", "earthquake": "ground",
        "rockslide": "rock", "bodyslam": "normal", "return": "normal", "doubleedge": "normal",
        "brickbreak": "fighting", "focuspunch": "fighting", "sludgebomb": "poison", "drillpeck": "flying",
        "hiddenpower": "normal", "crunch": "dark", "meteormash": "steel",
    }

    _KNOWN_RESPONSE_MIN = 0.08
    _SWITCH_MASS_FOR_IMMUNE_OVERRIDE = 0.60
    _SWITCH_MASS_FOR_WEAK_OVERRIDE = 0.45
    _WEAK_CURRENT_RATIO = 0.25
    _MODEL_SWITCH_HP_FLOOR = 0.30
    _MODEL_SWITCH_POISON_HP_FLOOR = 0.45
    _SWITCH_OVERRIDE_MARGIN = 4.0

    def __init__(self, opponent_model: OpponentModel | None = None, *, override_margin: float = 1.5):
        self.opponent_model = opponent_model or OpponentModel()
        self.override_margin = float(override_margin)

    @staticmethod
    def _move_id(move: Any) -> str:
        return str(getattr(move, "id", getattr(move, "name", "")) or "").lower().replace(" ", "").replace("-", "").replace("_", "")

    @classmethod
    def _move_type(cls, move: Any) -> str:
        value = getattr(move, "type", "")
        raw = str(getattr(value, "name", value) or "").lower().replace(" ", "").replace("-", "").replace("_", "")
        return raw or cls._MOVE_TYPES.get(cls._move_id(move), "")

    @staticmethod
    def _moves(battle: Any) -> list[Any]:
        active = getattr(battle, "active_pokemon", None)
        raw = list(getattr(active, "moves", {}).values()) if active is not None else []
        try:
            return list(consistent_move_order(raw))
        except ValueError:
            return sorted(raw, key=lambda m: str(getattr(m, "id", "")))

    @staticmethod
    def _switch_slots(battle: Any) -> list[Any]:
        raw = [p for p in (getattr(battle, "opponent_team", {}) or {}).values()
               if p is not None and not getattr(p, "fainted", False) and not getattr(p, "active", False)]
        try:
            return list(consistent_pokemon_order(raw))
        except ValueError:
            return sorted(raw, key=lambda p: str(getattr(p, "name", getattr(p, "species", ""))))

    @staticmethod
    def _own_switch_slots(battle: Any) -> list[Any]:
        raw = [p for p in (getattr(battle, "team", {}) or {}).values()
               if p is not None and not getattr(p, "fainted", False) and not getattr(p, "active", False)]
        try:
            return list(consistent_pokemon_order(raw))
        except ValueError:
            return sorted(raw, key=lambda p: str(getattr(p, "name", getattr(p, "species", ""))))

    @staticmethod
    def _matchup(attack_type: str, defender: Any) -> float:
        chart = {
            "fire": {"grass": 2, "ice": 2, "bug": 2, "steel": 2, "fire": .5, "water": .5, "rock": .5, "dragon": .5},
            "water": {"fire": 2, "ground": 2, "rock": 2, "water": .5, "grass": .5, "dragon": .5},
            "electric": {"water": 2, "flying": 2, "electric": .5, "grass": .5, "dragon": .5, "ground": 0},
            "grass": {"water": 2, "ground": 2, "rock": 2, "fire": .5, "grass": .5, "poison": .5, "flying": .5, "bug": .5, "dragon": .5, "steel": .5},
            "ice": {"grass": 2, "ground": 2, "flying": 2, "dragon": 2, "fire": .5, "water": .5, "ice": .5, "steel": .5},
            "fighting": {"normal": 2, "ice": 2, "rock": 2, "dark": 2, "steel": 2, "poison": .5, "flying": .5, "psychic": .5, "bug": .5, "ghost": 0},
            "ground": {"fire": 2, "electric": 2, "poison": 2, "rock": 2, "steel": 2, "grass": .5, "bug": .5, "flying": 0},
            "psychic": {"fighting": 2, "poison": 2, "steel": .5, "psychic": .5, "dark": 0},
            "rock": {"fire": 2, "ice": 2, "flying": 2, "bug": 2, "fighting": .5, "ground": .5, "steel": .5},
            "flying": {"grass": 2, "fighting": 2, "bug": 2, "electric": .5, "rock": .5, "steel": .5},
            "dark": {"psychic": 2, "ghost": 2, "fighting": .5, "dark": .5, "steel": .5},
            "ghost": {"ghost": 2, "psychic": 2, "dark": .5, "normal": 0},
            "steel": {"ice": 2, "rock": 2, "fire": .5, "water": .5, "electric": .5, "steel": .5},
        }
        multiplier = 1.0
        for defender_type in getattr(defender, "types", ()) or ():
            key = str(getattr(defender_type, "name", defender_type)).lower()
            multiplier *= chart.get(attack_type, {}).get(key, 1.0)
        return multiplier

    @staticmethod
    def _prediction_is_supported(battle: Any) -> bool:
        return getattr(battle, "opponent_active_pokemon", None) is not None

    def _current_damage(self, battle: Any, action: int) -> tuple[float, float, bool]:
        active = getattr(battle, "active_pokemon", None)
        target = getattr(battle, "opponent_active_pokemon", None)
        moves = self._moves(battle)
        if active is None or target is None or not (0 <= action < len(moves)):
            return 0.0, 0.0, False
        result = calculate_damage(active, target, moves[action], weather="")
        if not result.reliable:
            return 0.0, 0.0, False
        return float(result.percentage_max), float(result.ko_probability), True

    def _own_action_value(self, battle: Any, action: int, response: PredictedResponse) -> float:
        active = getattr(battle, "active_pokemon", None)
        target = getattr(battle, "opponent_active_pokemon", None)
        if active is None or target is None:
            return 0.0
        if action < 4:
            moves = self._moves(battle)
            if not (0 <= action < len(moves)):
                return -100.0
            move = moves[action]
            current = calculate_damage(active, target, move, weather="")
            if response.kind == "switch" and response.target:
                candidate = next((p for p in self._switch_slots(battle)
                                  if str(getattr(p, "species", "")).lower().replace(" ", "").replace("-", "").replace("_", "") == response.target), None)
                if candidate is not None:
                    branch = calculate_damage(active, candidate, move, weather="")
                    if branch.reliable:
                        return 0.80 * branch.percentage_max + 10.0 * branch.ko_probability
            if current.reliable:
                value = 0.18 * current.percentage_max + 12.0 * current.ko_probability
            else:
                value = 0.0
            if response.kind == "protect":
                value += 2.0 if self._move_id(move) in {"protect", "detect", "endure"} else -5.0
            elif response.kind == "setup":
                value += 3.0 * (current.ko_probability if current.reliable else 0.0)
            return value

        switches = self._own_switch_slots(battle)
        idx = action - 4
        if not (0 <= idx < len(switches)):
            return -100.0
        candidate = switches[idx]
        value = 2.0 + 5.0 * float(getattr(candidate, "current_hp_fraction", 1.0) or 1.0)
        if response.kind == "attack" and response.move:
            target_moves = list(getattr(target, "moves", {}).values())
            move_obj = next((m for m in target_moves if self._move_id(m) == response.move), None)
            if move_obj is not None:
                damage = calculate_damage(target, candidate, move_obj, weather="")
                if damage.reliable:
                    value += max(0.0, 18.0 - 0.20 * damage.percentage_max)
                    if damage.ko_probability >= 0.80:
                        value -= 18.0
                value += 5.0 * max(0.0, 1.0 - self._matchup(self._move_type(move_obj), candidate))
        return value

    def _model_switch_is_protected(self, battle: Any, model_action: int) -> bool:
        if model_action < 4:
            return False
        active = getattr(battle, "active_pokemon", None)
        if active is None:
            return False
        hp = float(getattr(active, "current_hp_fraction", 1.0) or 1.0)
        status = str(getattr(getattr(active, "status", None), "name", getattr(active, "status", "")) or "").lower()
        return hp <= self._MODEL_SWITCH_HP_FLOOR or (hp <= self._MODEL_SWITCH_POISON_HP_FLOOR and "poison" in status)

    def _predictive_attack_can_override_switch(self, battle: Any, action: int, responses: list[PredictedResponse], switch_mass: float) -> bool:
        if action >= 4:
            return False
        current_damage, current_ko, reliable = self._current_damage(battle, action)
        if not reliable:
            return False
        if current_ko >= 0.85:
            return True
        if switch_mass < 0.55:
            return False
        for response in responses:
            if response.kind != "switch" or response.probability < 0.22 or not response.target:
                continue
            candidate = next((p for p in self._switch_slots(battle)
                              if str(getattr(p, "species", "")).lower().replace(" ", "").replace("-", "").replace("_", "") == response.target), None)
            if candidate is None:
                continue
            moves = self._moves(battle)
            if action >= len(moves):
                continue
            branch = calculate_damage(getattr(battle, "active_pokemon"), candidate, moves[action], weather="")
            if branch.reliable and branch.ko_probability >= 0.80:
                return True
        return False

    def choose(self, battle: Any, legal_actions: list[int], model_action: int) -> tuple[int | None, str, list[ResponseScore]]:
        if not self._prediction_is_supported(battle):
            return None, "", []
        responses = self.opponent_model.predict_responses(battle)
        actionable = [r for r in responses if r.kind != "unknown" and r.probability >= self._KNOWN_RESPONSE_MIN]
        if not actionable:
            return None, "", []

        switch_mass = sum(r.probability for r in actionable if r.kind == "switch" and r.target)
        scores: list[ResponseScore] = []
        for action in legal_actions:
            expected = sum(response.probability * self._own_action_value(battle, int(action), response) for response in responses)
            current_damage, _, current_reliable = self._current_damage(battle, int(action))
            model_damage, _, model_reliable = self._current_damage(battle, int(model_action))

            if int(action) < 4 and current_reliable:
                if current_damage <= 0.0 and switch_mass < self._SWITCH_MASS_FOR_IMMUNE_OVERRIDE:
                    expected = -100.0
                if (int(action) != int(model_action) and int(model_action) < 4 and model_reliable
                        and current_damage < max(1.0, model_damage * self._WEAK_CURRENT_RATIO)
                        and switch_mass < self._SWITCH_MASS_FOR_WEAK_OVERRIDE):
                    expected = min(expected, model_damage * 0.10)

            anchor = 2.5 if int(action) == int(model_action) else 0.0
            scores.append(ResponseScore(int(action), expected + anchor, expected, ""))

        if self._model_switch_is_protected(battle, int(model_action)):
            best_attack = max((s for s in scores if s.action < 4), key=lambda s: s.score, default=None)
            if best_attack is None or not self._predictive_attack_can_override_switch(battle, best_attack.action, responses, switch_mass):
                return None, "", sorted(scores, key=lambda s: s.score, reverse=True)

        scores.sort(key=lambda x: x.score, reverse=True)
        best = scores[0]
        model = next((s for s in scores if s.action == int(model_action)), None)
        if model is None:
            return None, "", scores
        advantage = best.score - model.score
        required_margin = max(self.override_margin, self._SWITCH_OVERRIDE_MARGIN if int(model_action) >= 4 else self.override_margin)
        if best.action == model.action or advantage < required_margin:
            return None, "", scores

        top_responses = ", ".join(
            f"{r.kind}{'->'+r.target if r.target else (':'+r.move if r.move else '')} {r.probability:.0%}"
            for r in actionable
        )
        reason = f"response search: model={model.action} EV={model.expected:.2f}; predicted={top_responses}; best={best.action} EV={best.expected:.2f}; advantage={advantage:.2f}; canonical slots; re-evaluated this turn"
        return best.action, reason, scores
