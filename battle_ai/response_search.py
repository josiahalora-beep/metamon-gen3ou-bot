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


class ResponseSearcher:
    """Small expectiminimax-style evaluator over predicted opponent responses.

    This deliberately searches only the response branches supported by visible
    battle data and the opponent posterior. It does not assume hidden moves are
    known with certainty and it never replaces a strong model action for a tiny
    numerical advantage.
    """

    _MOVE_TYPES = {
        "surf": "water", "hydropump": "water", "icebeam": "ice", "thunderbolt": "electric",
        "thunder": "electric", "psychic": "psychic", "fireblast": "fire", "flamethrower": "fire",
        "gigadrain": "grass", "leafstorm": "grass", "energyball": "grass", "earthquake": "ground",
        "rockslide": "rock", "bodyslam": "normal", "return": "normal", "doubleedge": "normal",
        "brickbreak": "fighting", "focuspunch": "fighting", "sludgebomb": "poison", "drillpeck": "flying",
        "hiddenpower": "normal", "crunch": "dark", "meteor_mash": "steel", "meteormash": "steel",
    }

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
            own_damage = calculate_damage(active, target, move, weather="").percentage_max if calculate_damage(active, target, move, weather="").reliable else 0.0
            own_ko = calculate_damage(active, target, move, weather="").ko_probability if calculate_damage(active, target, move, weather="").reliable else 0.0
            value = 0.10 * own_damage + 8.0 * own_ko
            if response.kind == "switch" and response.target:
                candidate = next((p for p in self._switch_slots(battle) if str(getattr(p, "species", "")).lower().replace(" ", "").replace("-", "") == response.target), None)
                if candidate is not None:
                    result = calculate_damage(active, candidate, move, weather="")
                    if result.reliable:
                        value = 0.12 * result.percentage_max + 9.0 * result.ko_probability
                    move_type = self._move_type(move)
                    if move_type:
                        value += 3.0 * self._matchup(move_type, candidate)
            if response.kind == "protect":
                if self._move_id(move) in {"protect", "detect", "endure"}:
                    value += 2.0
                else:
                    value -= 6.0
            if response.kind == "setup":
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
        return survival_bonus + 4.0 * max(0.0, 1.0 - self._matchup(response.move and self._move_type(next((m for m in (getattr(target, "moves", {}) or {}).values() if self._move_id(m) == response.move), None)) or "", candidate)) if response.kind == "attack" else survival_bonus

    def choose(self, battle: Any, legal_actions: list[int], model_action: int) -> tuple[int | None, str, list[ResponseScore]]:
        responses = self.opponent_model.predict_responses(battle)
        if not responses:
            return None, "", []

        scores: list[ResponseScore] = []
        for action in legal_actions:
            expected = 0.0
            branches = []
            for response in responses:
                branch = self._own_action_value(battle, int(action), response)
                expected += response.probability * branch
                branches.append((response.kind, response.probability, branch))
            # Keep the learned model anchored while permitting predictive moves
            # to win when they outperform it across the posterior, not a single
            # cherry-picked branch.
            anchor = 1.25 if int(action) == int(model_action) else 0.0
            rationale = "; ".join(f"{k}={p:.0%}:{v:.1f}" for k, p, v in branches if p >= 0.05)
            scores.append(ResponseScore(int(action), expected + anchor, expected, rationale))

        scores.sort(key=lambda x: x.score, reverse=True)
        best = scores[0]
        model = next((s for s in scores if s.action == int(model_action)), None)
        if model is None:
            return best.action, "response search chose highest expected value legal action", scores
        advantage = best.score - model.score
        if best.action == model.action or advantage < self.override_margin:
            return None, "", scores

        top_responses = ", ".join(
            f"{r.kind}{'->'+r.target if r.target else (':'+r.move if r.move else '')} {r.probability:.0%}"
            for r in responses if r.probability >= 0.08
        )
        reason = f"response search: model={model.action} EV={model.expected:.2f}; predicted={top_responses}; best={best.action} EV={best.expected:.2f}; advantage={advantage:.2f}; re-evaluated this turn"
        return best.action, reason, scores
