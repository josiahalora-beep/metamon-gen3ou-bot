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
    """Conservative bounded response search layered under the learned policy.

    The search may improve a damaging move using a well-supported response
    prediction. It cannot casually convert switches, setup, passive moves, or
    self-sacrificial attacks into speculative trades.
    """

    _KNOWN_RESPONSE_MIN = 0.08
    _STRONG_SWITCH_PROB = 0.55
    _SPECIFIC_SWITCH_PROB = 0.18
    _MOVE_OVERRIDE_DAMAGE_GAIN = 20.0
    _MOVE_OVERRIDE_KO_GAIN = 0.20
    _MODEL_SWITCH_HP_FLOOR = 0.30
    _MODEL_SWITCH_POISON_HP_FLOOR = 0.45
    _SWITCH_OVERRIDE_MARGIN = 5.0

    _SELF_KO_MOVES = {"explosion", "selfdestruct", "memento"}

    def __init__(self, opponent_model: OpponentModel | None = None, *, override_margin: float = 1.5):
        self.opponent_model = opponent_model or OpponentModel()
        self.override_margin = float(override_margin)

    @staticmethod
    def _move_id(move: Any) -> str:
        return str(getattr(move, "id", getattr(move, "name", "")) or "").lower().replace(" ", "").replace("-", "").replace("_", "")

    @staticmethod
    def _move_power(move: Any) -> int:
        return int(getattr(move, "base_power", 0) or 0)

    @staticmethod
    def _moves(battle: Any) -> list[Any]:
        active = getattr(battle, "active_pokemon", None)
        raw = list(getattr(active, "moves", {}).values()) if active is not None else []
        try:
            return list(consistent_move_order(raw))
        except ValueError:
            return sorted(raw, key=lambda m: str(getattr(m, "id", "")))

    @staticmethod
    def _own_switch_slots(battle: Any) -> list[Any]:
        raw = [p for p in (getattr(battle, "team", {}) or {}).values()
               if p is not None and not getattr(p, "fainted", False) and not getattr(p, "active", False)]
        try:
            return list(consistent_pokemon_order(raw))
        except ValueError:
            return sorted(raw, key=lambda p: str(getattr(p, "name", getattr(p, "species", ""))))

    @staticmethod
    def _opponent_switch_slots(battle: Any) -> list[Any]:
        raw = [p for p in (getattr(battle, "opponent_team", {}) or {}).values()
               if p is not None and not getattr(p, "fainted", False) and not getattr(p, "active", False)]
        try:
            return list(consistent_pokemon_order(raw))
        except ValueError:
            return sorted(raw, key=lambda p: str(getattr(p, "name", getattr(p, "species", ""))))

    @staticmethod
    def _normalize_species(pokemon: Any) -> str:
        return str(getattr(pokemon, "species", getattr(pokemon, "name", "")) or "").lower().replace(" ", "").replace("-", "").replace("_", "")

    @staticmethod
    def _status_name(pokemon: Any) -> str:
        value = getattr(pokemon, "status", "")
        return str(getattr(value, "name", value) or "").lower()

    @staticmethod
    def _damage_percent(result: Any) -> float:
        # Damage reports can be expressed relative to current HP. That is
        # useful diagnostically but disastrous as an EV term at 1% HP.
        return min(100.0, max(0.0, float(getattr(result, "percentage_max", 0.0) or 0.0)))

    @staticmethod
    def _move_type(move: Any) -> str:
        value = getattr(move, "type", "")
        return str(getattr(value, "name", value) or "").lower().replace(" ", "").replace("-", "").replace("_", "")

    def _prediction_is_supported(self, battle: Any) -> bool:
        return getattr(battle, "opponent_active_pokemon", None) is not None

    def _current_damage(self, battle: Any, action: int) -> tuple[float, float, bool]:
        active = getattr(battle, "active_pokemon", None)
        target = getattr(battle, "opponent_active_pokemon", None)
        moves = self._moves(battle)
        if active is None or target is None or not (0 <= action < len(moves)):
            return 0.0, 0.0, False
        move = moves[action]
        result = calculate_damage(active, target, move, weather="")
        if not result.reliable:
            return 0.0, 0.0, False
        return self._damage_percent(result), float(getattr(result, "ko_probability", 0.0) or 0.0), True

    def _predicted_switch(self, battle: Any, response: PredictedResponse) -> Any | None:
        target = response.target
        if not target:
            return None
        return next((p for p in self._opponent_switch_slots(battle)
                     if self._normalize_species(p) == target), None)

    def _expected_move_value(self, battle: Any, action: int, responses: list[PredictedResponse]) -> float:
        active = getattr(battle, "active_pokemon", None)
        target = getattr(battle, "opponent_active_pokemon", None)
        moves = self._moves(battle)
        if active is None or target is None or not (0 <= action < len(moves)):
            return -100.0

        move = moves[action]
        if self._move_power(move) <= 0:
            return 0.0

        current = calculate_damage(active, target, move, weather="")
        current_damage = self._damage_percent(current) if current.reliable else 0.0
        current_ko = float(getattr(current, "ko_probability", 0.0) or 0.0) if current.reliable else 0.0
        value = 0.20 * current_damage + 12.0 * current_ko

        for response in responses:
            if response.kind != "switch" or not response.target or response.probability < self._SPECIFIC_SWITCH_PROB:
                continue
            candidate = self._predicted_switch(battle, response)
            if candidate is None:
                continue
            branch = calculate_damage(active, candidate, move, weather="")
            if not branch.reliable:
                continue
            branch_damage = self._damage_percent(branch)
            branch_ko = float(getattr(branch, "ko_probability", 0.0) or 0.0)
            branch_value = 0.20 * branch_damage + 12.0 * branch_ko
            # Only credit a portion of an uncertain future switch.
            value += response.probability * max(0.0, branch_value - value) * 0.65

        return value

    def _expected_switch_value(self, battle: Any, action: int, responses: list[PredictedResponse]) -> float:
        slots = self._own_switch_slots(battle)
        idx = action - 4
        if not (0 <= idx < len(slots)):
            return -100.0
        candidate = slots[idx]
        hp = float(getattr(candidate, "current_hp_fraction", 1.0) or 1.0)
        value = 5.0 + 8.0 * hp
        opponent = getattr(battle, "opponent_active_pokemon", None)
        for response in responses:
            if response.kind != "attack" or not response.move or response.probability < self._KNOWN_RESPONSE_MIN:
                continue
            move_obj = next((m for m in (getattr(opponent, "moves", {}) or {}).values()
                             if self._move_id(m) == response.move), None)
            if move_obj is None:
                continue
            damage = calculate_damage(opponent, candidate, move_obj, weather="")
            if not damage.reliable:
                continue
            dmg = self._damage_percent(damage)
            value += response.probability * max(-20.0, 12.0 - 0.18 * dmg)
            if float(getattr(damage, "ko_probability", 0.0) or 0.0) >= 0.8:
                value -= 15.0 * response.probability
        return value

    def _model_switch_is_protected(self, battle: Any, model_action: int) -> bool:
        if model_action < 4:
            return False
        active = getattr(battle, "active_pokemon", None)
        if active is None:
            return False
        hp = float(getattr(active, "current_hp_fraction", 1.0) or 1.0)
        status = self._status_name(active)
        return hp <= self._MODEL_SWITCH_HP_FLOOR or (hp <= self._MODEL_SWITCH_POISON_HP_FLOOR and "poison" in status)

    def _can_replace_switch_with_attack(self, battle: Any, action: int, responses: list[PredictedResponse], switch_mass: float) -> bool:
        moves = self._moves(battle)
        if not (0 <= action < len(moves)):
            return False
        move = moves[action]
        if self._move_power(move) <= 0 or self._move_id(move) in self._SELF_KO_MOVES:
            return False

        _, current_ko, reliable = self._current_damage(battle, action)
        if not reliable:
            return False

        # A normal immediate KO can justify staying in. A self-KO never can
        # unless higher-level strategic logic explicitly selected it.
        if current_ko >= 0.85:
            return True

        if switch_mass < self._STRONG_SWITCH_PROB:
            return False

        for response in responses:
            if response.kind != "switch" or response.probability < self._STRONG_SWITCH_PROB or not response.target:
                continue
            candidate = self._predicted_switch(battle, response)
            if candidate is None:
                continue
            branch = calculate_damage(getattr(battle, "active_pokemon", None), candidate, move, weather="")
            if branch.reliable and float(getattr(branch, "ko_probability", 0.0) or 0.0) >= 0.80:
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

        # Prediction should refine attack-vs-attack choices. It should not
        # invent a switch when the learned policy chose to attack.
        if model_action < 4:
            model_moves = self._moves(battle)
            model_is_damaging = 0 <= model_action < len(model_moves) and self._move_power(model_moves[model_action]) > 0
            for action in legal_actions:
                action = int(action)
                if action >= 4:
                    scores.append(ResponseScore(action, -50.0, -50.0, "prediction layer does not create speculative switches"))
                    continue
                moves = self._moves(battle)
                if not (0 <= action < len(moves)):
                    scores.append(ResponseScore(action, -100.0, -100.0, "invalid move slot"))
                    continue
                move = moves[action]
                if self._move_power(move) <= 0:
                    # Preserve setup/passive decisions for strategic layers.
                    expected = 0.0 if action == model_action else -2.0
                else:
                    expected = self._expected_move_value(battle, action, responses)
                if action != model_action and model_is_damaging:
                    model_value = self._expected_move_value(battle, model_action, responses)
                    if expected - model_value < self._MOVE_OVERRIDE_DAMAGE_GAIN / 5.0:
                        expected = min(expected, model_value + 1.0)
                elif action != model_action and not model_is_damaging:
                    expected = -2.0
                anchor = 2.5 if action == model_action else 0.0
                scores.append(ResponseScore(action, expected + anchor, expected, ""))
        else:
            # Learned policy chose a switch. Only a mechanically strong attack
            # may overturn it, and never a self-sacrificial move.
            for action in legal_actions:
                action = int(action)
                if action < 4:
                    move_value = self._expected_move_value(battle, action, responses)
                    scores.append(ResponseScore(action, move_value, move_value, "attack candidate against protected switch"))
                else:
                    switch_value = self._expected_switch_value(battle, action, responses)
                    anchor = 3.0 if action == model_action else 0.0
                    scores.append(ResponseScore(action, switch_value + anchor, switch_value, "switch-preservation baseline"))

        scores.sort(key=lambda s: s.score, reverse=True)
        model = next((s for s in scores if s.action == int(model_action)), None)
        if model is None:
            return None, "", scores
        best = scores[0]

        if best.action != model.action:
            if model_action >= 4:
                if self._model_switch_is_protected(battle, model_action) and not self._can_replace_switch_with_attack(battle, best.action, responses, switch_mass):
                    return None, "", scores
                if best.action < 4:
                    moves = self._moves(battle)
                    if best.action >= len(moves) or self._move_id(moves[best.action]) in self._SELF_KO_MOVES:
                        return None, "", scores
                    if not self._can_replace_switch_with_attack(battle, best.action, responses, switch_mass):
                        return None, "", scores
            else:
                # Never let prediction turn a setup/passive move into a random
                # damaging move. The strategic planner owns those transitions.
                moves = self._moves(battle)
                if not (0 <= model_action < len(moves)) or self._move_power(moves[model_action]) <= 0:
                    return None, "", scores

                model_damage, model_ko, model_ok = self._current_damage(battle, model_action)
                best_damage, best_ko, best_ok = self._current_damage(battle, best.action)
                material = (
                    best_ok and model_ok and (
                        best_ko >= 0.85 and model_ko < 0.85
                        or best_ko - model_ko >= self._MOVE_OVERRIDE_KO_GAIN
                        or best_damage - model_damage >= self._MOVE_OVERRIDE_DAMAGE_GAIN
                    )
                )
                if not material:
                    return None, "", scores

        advantage = best.score - model.score
        required_margin = self._SWITCH_OVERRIDE_MARGIN if model_action >= 4 else max(self.override_margin, 1.5)
        if best.action == model.action or advantage < required_margin:
            return None, "", scores

        top = ", ".join(
            f"{r.kind}{'->'+r.target if r.target else (':'+r.move if r.move else '')} {r.probability:.0%}"
            for r in actionable
        )
        reason = (
            f"response search: model={model.action} EV={model.expected:.2f}; "
            f"predicted={top}; best={best.action} EV={best.expected:.2f}; "
            f"advantage={advantage:.2f}; canonical slots; re-evaluated this turn"
        )
        return best.action, reason, scores
