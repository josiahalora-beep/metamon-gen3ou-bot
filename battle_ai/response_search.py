from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .damage import calculate_damage
from .opponent_model import OpponentModel, PredictedResponse


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

    Prediction may enhance a move, but hidden or weakly-supported future
    branches cannot make an objectively bad current-board move preferable.
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
    _SWITCH_MASS_FOR_IMMUNE_OVERRIDE = 0.50
    _SWITCH_MASS_FOR_WEAK_OVERRIDE = 0.35
    _WEAK_CURRENT_RATIO = 0.25
    _MODEL_SWITCH_HP_FLOOR = 0.30
    _MODEL_SWITCH_POISON_HP_FLOOR = 0.45
    _SWITCH_OVERRIDE_MARGIN = 4.0

    def __init__(self, opponent_model: OpponentModel | None = None, *, override_margin: float = 1.5):
        self.opponent_model = opponent_model or OpponentModel()
        self.override_margin = float(override_margin)

    @staticmethod
    def _move_id(move: Any) -> str:
        return str(getattr(move, "id", getattr(move, "name", "")) or "").lower().replace(" ", "").replace("-", "")

    @classmethod
    def _move_type(cls, move: Any) -> str:
        value = getattr(move, "type", "")
        raw = str(getattr(value, "name", value) or "").lower().replace(" ", "").replace("-", "")
        return raw or cls._MOVE_TYPES.get(cls._move_id(move), "")

    @staticmethod
    def _switch_slots(battle: Any) -> list[Any]:
        return [p for p in (getattr(battle, "opponent_team", {}) or {}).values()
                if p is not None and not getattr(p, "fainted", False) and not getattr(p, "active", False)]

    @staticmethod
    def _own_switch_slots(battle: Any) -> list[Any]:
        return [p for p in (getattr(battle, "team", {}) or {}).values()
                if p is not None and not getattr(p, "fainted", False) and not getattr(p, "active", False)]

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
        opponent = getattr(battle, "opponent_active_pokemon", None)
        if opponent is None:
            return False
        opponent_team = getattr(battle, "opponent_team", {}) or {}
        if not opponent_team:
            stats = getattr(opponent, "stats", {}) or {}
            base_stats = getattr(opponent, "base_stats", {}) or {}
            revealed_moves = getattr(opponent, "moves", {}) or {}
            has_known_stats = any(v is not None for v in stats.values()) if isinstance(stats, dict) else False
            has_species_stats = any(v is not None for v in base_stats.values()) if isinstance(base_stats, dict) and base_stats else False
            if not has_known_stats and not has_species_stats and not revealed_moves:
                return False
        return True

    def _current_damage(self, battle: Any, action: int) -> tuple[float, float, bool]:
        active = getattr(battle, "active_pokemon", None)
        target = getattr(battle, "opponent_active_pokemon", None)
        if active is None or target is None or action >= 4:
            return 0.0, 0.0, False
        moves = list(getattr(active, "moves", {}).values())
        if not (0 <= action < len(moves)):
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
            moves = list(getattr(active, "moves", {}).values())
            if action >= len(moves):
                return -100.0
            move = moves[action]
            result = calculate_damage(active, target, move, weather="")
            own_damage = result.percentage_max if result.reliable else 0.0
            own_ko = result.ko_probability if result.reliable else 0.0
            value = 0.10 * own_damage + 8.0 * own_ko
            if response.kind == "switch" and response.target:
                candidate = next((p for p in self._switch_slots(battle)
                                  if str(getattr(p, "species", "")).lower().replace(" ", "").replace("-", "") == response.target), None)
                if candidate is not None:
                    branch = calculate_damage(active, candidate, move, weather="")
                    if branch.reliable:
                        value = 0.12 * branch.percentage_max + 9.0 * branch.ko_probability
                    move_type = self._move_type(move)
                    if move_type:
                        value += 3.0 * self._matchup(move_type, candidate)
            elif response.kind == "protect":
                value += 2.0 if self._move_id(move) in {"protect", "detect", "endure"} else -6.0
            elif response.kind == "setup":
                value += 4.0 * own_ko
            return value

        switches = self._own_switch_slots(battle)
        idx = action - 4
        if idx >= len(switches):
            return -100.0
        candidate = switches[idx]
        survival_bonus = 5.0 * float(getattr(candidate, "current_hp_fraction", 1.0) or 1.0)
        if response.kind == "attack" and response.move:
            move_obj = next((m for m in (getattr(target, "moves", {}) or {}).values() if self._move_id(m) == response.move), None)
            if move_obj is not None:
                damage = calculate_damage(target, candidate, move_obj, weather="")
                if damage.reliable:
                    survival_bonus += max(0.0, 20.0 - damage.percentage_max * 0.15)
                survival_bonus += 4.0 * max(0.0, 1.0 - self._matchup(self._move_type(move_obj), candidate))
        return survival_bonus

    def _model_switch_is_protected(self, battle: Any, model_action: int) -> bool:
        if model_action < 4:
            return False
        active = getattr(battle, "active_pokemon", None)
        if active is None:
            return False
        hp = float(getattr(active, "current_hp_fraction", 1.0) or 1.0)
        status = str(getattr(getattr(active, "status", None), "name", getattr(active, "status", "")) or "").lower()
        # Once the policy has elected to preserve a genuinely endangered
        # Pokémon, prediction must not casually force it back into combat.
        if hp <= self._MODEL_SWITCH_HP_FLOOR:
            return True
        if hp <= self._MODEL_SWITCH_POISON_HP_FLOOR and "poison" in status:
            return True
        return False

    def _predictive_attack_can_override_switch(self, battle: Any, action: int, responses: list[PredictedResponse], switch_mass: float) -> bool:
        if action >= 4:
            return False
        current_damage, current_ko, reliable = self._current_damage(battle, action)
        if not reliable:
            return False
        # A switch-punishing attack needs either a real immediate KO or a very
        # strong, specific switch posterior. Ordinary 20–30% switch guesses do
        # not justify abandoning the learned switch decision.
        if current_ko >= 0.85:
            return True
        if switch_mass < 0.45:
            return False
        for response in responses:
            if response.kind != "switch" or response.probability < 0.18 or not response.target:
                continue
            candidate = next((p for p in self._switch_slots(battle)
                              if str(getattr(p, "species", "")).lower().replace(" ", "").replace("-", "") == response.target), None)
            if candidate is None:
                continue
            active = getattr(battle, "active_pokemon", None)
            if active is None:
                continue
            moves = list(getattr(active, "moves", {}).values())
            if action >= len(moves):
                continue
            branch = calculate_damage(active, candidate, moves[action], weather="")
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
            expected = 0.0
            branches = []
            for response in responses:
                branch = self._own_action_value(battle, int(action), response)
                expected += response.probability * branch
                branches.append((response.kind, response.probability, branch))

            current_damage, _, current_reliable = self._current_damage(battle, int(action))
            model_damage, _, model_reliable = self._current_damage(battle, int(model_action))

            if int(action) < 4 and current_reliable:
                if current_damage <= 0.0:
                    if switch_mass < self._SWITCH_MASS_FOR_IMMUNE_OVERRIDE or not any(r.target for r in actionable if r.kind == "switch"):
                        expected = min(expected, 0.0)
                if (int(action) != int(model_action) and model_action < 4 and model_reliable
                        and current_damage < model_damage * self._WEAK_CURRENT_RATIO
                        and switch_mass < self._SWITCH_MASS_FOR_WEAK_OVERRIDE):
                    expected = min(expected, model_damage * 0.10)

            anchor = 1.25 if int(action) == int(model_action) else 0.0
            rationale = "; ".join(f"{k}={p:.0%}:{v:.1f}" for k, p, v in branches if p >= 0.05)
            scores.append(ResponseScore(int(action), expected + anchor, expected, rationale))

        scores.sort(key=lambda x: x.score, reverse=True)
        best = scores[0]
        model = next((s for s in scores if s.action == int(model_action)), None)
        if model is None:
            return best.action, "response search chose highest expected value legal action", scores

        # Critical policy protection: when the learned model says SWITCH with
        # an endangered/poisoned active, response search cannot repeatedly talk
        # it back into combat unless there is an immediate KO or high-confidence
        # switch punishment that is worth the risk.
        if self._model_switch_is_protected(battle, int(model_action)):
            if not self._predictive_attack_can_override_switch(battle, int(best.action), responses, switch_mass):
                return None, "", scores

        advantage = best.score - model.score
        required_margin = self.override_margin
        if int(model_action) >= 4:
            required_margin = max(required_margin, self._SWITCH_OVERRIDE_MARGIN)
        if best.action == model.action or advantage < required_margin:
            return None, "", scores

        top_responses = ", ".join(
            f"{r.kind}{'->'+r.target if r.target else (':'+r.move if r.move else '')} {r.probability:.0%}"
            for r in actionable
        )
        reason = f"response search: model={model.action} EV={model.expected:.2f}; predicted={top_responses}; best={best.action} EV={best.expected:.2f}; advantage={advantage:.2f}; re-evaluated this turn"
        return best.action, reason, scores
