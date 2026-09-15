from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .smogon_priors import likely_moves, load_profiles


def _key(value: Any) -> str:
    return str(getattr(value, "id", getattr(value, "name", value)) or "").lower().replace(" ", "").replace("-", "").replace("_", "")


def _species(value: Any) -> str:
    return str(getattr(value, "species", getattr(value, "name", value)) or "").lower().replace(" ", "").replace("-", "").replace("_", "")


def _revealed_moves(pokemon: Any) -> tuple[str, ...]:
    return tuple(_key(m) for m in (getattr(pokemon, "moves", {}) or {}).values())


def _base_stats(pokemon: Any) -> dict[str, int]:
    return {str(k).lower(): int(v) for k, v in (getattr(pokemon, "base_stats", {}) or {}).items() if v is not None}


@dataclass(frozen=True)
class PredictedResponse:
    kind: str
    probability: float
    target: str = ""
    move: str = ""
    rationale: str = ""


@dataclass
class _BehaviorState:
    switches: int = 0
    opportunities: int = 0
    attacks: int = 0
    setup_uses: int = 0
    protect_uses: int = 0
    passive_uses: int = 0

    @property
    def observations(self) -> int:
        return self.attacks + self.setup_uses + self.protect_uses + self.passive_uses

    def switch_rate(self) -> float:
        # Conservative Beta-style prior: 2/10 switch tendency before evidence.
        return (self.switches + 2.0) / (self.opportunities + 10.0)

    def attack_rate(self) -> float:
        return (self.attacks + 4.0) / (self.observations + 8.0)


class OpponentModel:
    """Visible-information Bayesian opponent model for Gen 3 OU.

    Population evidence comes from the Gen 3 set corpus. Match-specific
    behavior is updated only when the observed state changes, preventing the
    same turn from being counted repeatedly. Unknown information stays unknown.
    """

    _MOVE_TYPES = {
        "surf": "water", "hydropump": "water", "icebeam": "ice", "thunderbolt": "electric",
        "thunder": "electric", "psychic": "psychic", "fireblast": "fire", "flamethrower": "fire",
        "gigadrain": "grass", "leafstorm": "grass", "energyball": "grass", "earthquake": "ground",
        "rockslide": "rock", "bodyslam": "normal", "return": "normal", "doubleedge": "normal",
        "brickbreak": "fighting", "focuspunch": "fighting", "sludgebomb": "poison", "drillpeck": "flying",
        "hiddenpower": "normal", "crunch": "dark", "meteormash": "steel", "meteor_mash": "steel",
    }

    _SETUP = {"swordsdance", "dragondance", "calmmind", "curse", "agility", "rockpolish", "bellydrum", "growth", "amnesia", "irondefense"}
    _PASSIVE = {"toxic", "thunderwave", "willowisp", "leechseed", "roar", "whirlwind", "haze", "spikes", "rapidspin"}
    _PROTECT = {"protect", "detect", "endure"}

    def __init__(self, *, prior_strength: float = 8.0):
        self.prior_strength = float(prior_strength)
        self._behavior: dict[str, _BehaviorState] = defaultdict(_BehaviorState)
        self._previous: dict[str, tuple[str, tuple[str, ...], float]] = {}

    def reset(self, battle_id: str) -> None:
        self._behavior.pop(str(battle_id), None)
        self._previous.pop(str(battle_id), None)

    @classmethod
    def move_type(cls, move_id: str) -> str:
        return cls._MOVE_TYPES.get(_key(move_id), "")

    @staticmethod
    def _move_class(move: str) -> str:
        move = _key(move)
        if move in OpponentModel._PROTECT:
            return "protect"
        if move in OpponentModel._SETUP:
            return "setup"
        if move in OpponentModel._PASSIVE:
            return "passive"
        return "attack"

    def observe_transition(self, battle: Any) -> None:
        battle_id = str(getattr(battle, "battle_tag", "unknown"))
        active = getattr(battle, "opponent_active_pokemon", None)
        if active is None:
            return
        current_species = _species(active)
        current_moves = _revealed_moves(active)
        hp_fraction = float(getattr(active, "current_hp_fraction", 1.0) or 1.0)
        current = (current_species, current_moves, round(hp_fraction, 3))
        previous = self._previous.get(battle_id)
        if previous == current:
            return
        state = self._behavior[battle_id]
        if previous is not None:
            previous_species, previous_moves, previous_hp = previous
            state.opportunities += 1
            if current_species != previous_species:
                state.switches += 1
            else:
                newly_revealed = set(current_moves) - set(previous_moves)
                for move in newly_revealed:
                    if move in self._PROTECT:
                        state.protect_uses += 1
                    elif move in self._SETUP:
                        state.setup_uses += 1
                    elif move in self._PASSIVE:
                        state.passive_uses += 1
                    else:
                        state.attacks += 1
                # HP loss alone is not treated as an attack: residual damage,
                # poison, weather, recoil and hazards are all possible causes.
        self._previous[battle_id] = current

    def _profile_move_prior(self, pokemon: Any) -> dict[str, float]:
        species = str(getattr(pokemon, "species", getattr(pokemon, "name", "")))
        observed = _revealed_moves(pokemon)
        profiles = load_profiles(species, revealed_moves=observed)
        weights: dict[str, float] = defaultdict(float)
        if profiles:
            profile_weights = [max(0.0, p.weight) for p in profiles[:12]]
            total = sum(profile_weights)
            for profile, profile_weight in zip(profiles[:12], profile_weights):
                share = profile_weight / max(total, 1e-9)
                for move in profile.moves:
                    weights[_key(move)] += share / max(1, len(profile.moves))
        else:
            for index, move in enumerate(likely_moves(pokemon, limit=8)):
                weights[_key(move)] += 1.0 / (index + 1)
        for move in observed:
            weights[move] = max(weights.get(move, 0.0), 0.50)
        total = sum(weights.values())
        return {move: value / total for move, value in weights.items()} if total > 0 else {}

    @staticmethod
    def _effectiveness(move_type: str, defender_types: set[str]) -> float:
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
            "poison": {"grass": 2, "poison": .5, "ground": .5, "rock": .5, "ghost": .5, "steel": 0},
        }
        result = 1.0
        for defender_type in defender_types:
            result *= chart.get(move_type, {}).get(defender_type, 1.0)
        return result

    def _switch_probability(self, battle: Any, behavior: _BehaviorState) -> float:
        opponent = getattr(battle, "opponent_active_pokemon", None)
        our_active = getattr(battle, "active_pokemon", None)
        prior = 0.18
        if opponent is not None and float(getattr(opponent, "current_hp_fraction", 1.0) or 1.0) <= 0.30:
            prior += 0.10
        if opponent is not None and our_active is not None:
            opp_types = {_key(t) for t in (getattr(opponent, "types", ()) or ())}
            our_types = {_key(t) for t in (getattr(our_active, "types", ()) or ())}
            # Type pressure is only a weak prior; it never creates certainty.
            if {"ground", "rock", "flying", "water", "fire", "ice", "fighting"} & our_types and opp_types & {"fire", "ice", "bug", "steel", "rock", "water", "ground"}:
                prior += 0.04
        behavioral = behavior.switch_rate()
        n = min(behavior.opportunities, 12)
        probability = (prior * self.prior_strength + behavioral * n) / (self.prior_strength + n)
        return min(0.55, max(0.10, probability))

    def predict_responses(self, battle: Any, *, max_switches: int = 3, max_moves: int = 4) -> list[PredictedResponse]:
        self.observe_transition(battle)
        opponent = getattr(battle, "opponent_active_pokemon", None)
        if opponent is None:
            return []
        battle_id = str(getattr(battle, "battle_tag", "unknown"))
        behavior = self._behavior[battle_id]
        move_prior = self._profile_move_prior(opponent)
        responses: list[PredictedResponse] = []

        switch_prob = self._switch_probability(battle, behavior)
        candidates: list[tuple[float, Any]] = []
        for p in (getattr(battle, "opponent_team", {}) or {}).values():
            if p is None or p is opponent or getattr(p, "fainted", False) or getattr(p, "active", False):
                continue
            p_types = {_key(t) for t in (getattr(p, "types", ()) or ())}
            score = 1.0 * float(getattr(p, "current_hp_fraction", 1.0) or 1.0)
            resisted = 0.0
            punished = 0.0
            for move_id, move_prob in move_prior.items():
                move_type = self.move_type(move_id)
                if not move_type:
                    continue
                eff = self._effectiveness(move_type, p_types)
                if eff < 1.0:
                    resisted += move_prob * (1.0 - eff)
                elif eff > 1.0:
                    punished += move_prob * (eff - 1.0)
            score += 1.2 * resisted - 0.6 * punished
            candidates.append((max(0.05, score), p))
        candidates.sort(key=lambda item: item[0], reverse=True)
        top_switches = candidates[:max_switches]
        if top_switches:
            denom = sum(score for score, _ in top_switches)
            for score, p in top_switches:
                probability = switch_prob * score / max(denom, 1e-9)
                responses.append(PredictedResponse("switch", probability, target=_species(p), rationale="bounded switch prior + set-based matchup posterior"))

        remaining = max(0.0, 1.0 - switch_prob)
        attacks = sorted(move_prior.items(), key=lambda kv: kv[1], reverse=True)[:max_moves]
        if attacks:
            total = sum(prob for _, prob in attacks)
            behavior_attack = behavior.attack_rate()
            for move, prob in attacks:
                share = prob / max(total, 1e-9)
                adjusted = remaining * (0.75 * share + 0.25 * behavior_attack * share)
                responses.append(PredictedResponse(self._move_class(move), adjusted, move=move, rationale="Gen 3 set posterior + revealed moves + smoothed behavior"))

        total = sum(max(0.0, r.probability) for r in responses)
        if total < 0.995:
            responses.append(PredictedResponse("unknown", 1.0 - total, rationale="unresolved hidden-information mass"))
        elif total > 1.005:
            responses = [PredictedResponse(r.kind, r.probability / total, r.target, r.move, r.rationale) for r in responses]
        return responses
