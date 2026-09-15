from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import exp
from typing import Any

from .smogon_priors import infer_profile, likely_moves, load_profiles


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
    switch_opportunities: int = 0
    attacks: int = 0
    setup_uses: int = 0
    protect_uses: int = 0
    passive_uses: int = 0

    def smoothed_switch_rate(self) -> float:
        return (self.switches + 1.5) / (self.switch_opportunities + 3.0)

    def smoothed_attack_rate(self) -> float:
        total = self.attacks + self.setup_uses + self.protect_uses + self.passive_uses
        return (self.attacks + 1.0) / (total + 2.0)


class OpponentModel:
    """Per-battle Bayesian behavior model layered over population priors.

    Population beliefs come from the Gen 3 set corpus. In-match behavior is
    updated with Laplace-smoothed evidence so three observations cannot
    overwhelm the population prior. State is keyed by battle id and naturally
    re-evaluated every turn.
    """

    def __init__(self, *, prior_strength: float = 8.0):
        self.prior_strength = float(prior_strength)
        self._behavior: dict[str, _BehaviorState] = defaultdict(_BehaviorState)
        self._previous: dict[str, tuple[str, tuple[str, ...], float]] = {}

    def reset(self, battle_id: str) -> None:
        self._behavior.pop(str(battle_id), None)
        self._previous.pop(str(battle_id), None)

    def observe_transition(self, battle: Any) -> None:
        battle_id = str(getattr(battle, "battle_tag", "unknown"))
        active = getattr(battle, "opponent_active_pokemon", None)
        if active is None:
            return
        current_species = _species(active)
        current_moves = _revealed_moves(active)
        hp_fraction = float(getattr(active, "current_hp_fraction", 1.0) or 1.0)
        previous = self._previous.get(battle_id)
        if previous is None:
            self._previous[battle_id] = (current_species, current_moves, hp_fraction)
            return

        previous_species, previous_moves, previous_hp = previous
        state = self._behavior[battle_id]
        state.switch_opportunities += 1
        if current_species != previous_species:
            state.switches += 1
        else:
            new_moves = set(current_moves) - set(previous_moves)
            if new_moves:
                for move in new_moves:
                    if move in {"protect", "detect", "endure"}:
                        state.protect_uses += 1
                    elif move in {
                        "swordsdance", "dragondance", "calmmind", "curse", "agility",
                        "rockpolish", "bellydrum", "growth", "amnesia", "irondefense",
                    }:
                        state.setup_uses += 1
                    elif move in {"leechseed", "toxic", "thunderwave", "willowisp", "roar", "whirlwind", "haze", "spikes"}:
                        state.passive_uses += 1
                    else:
                        state.attacks += 1
            elif hp_fraction < previous_hp - 0.01:
                state.attacks += 1

        self._previous[battle_id] = (current_species, current_moves, hp_fraction)

    def _profile_move_prior(self, pokemon: Any) -> dict[str, float]:
        species = str(getattr(pokemon, "species", getattr(pokemon, "name", "")))
        observed = _revealed_moves(pokemon)
        profiles = load_profiles(species, revealed_moves=observed)
        weights: dict[str, float] = defaultdict(float)
        if profiles:
            total = sum(max(0.0, p.weight) for p in profiles)
            for profile in profiles[:12]:
                share = max(0.0, profile.weight) / max(total, 1e-9)
                for move in profile.moves:
                    weights[_key(move)] += share / max(1, len(profile.moves))
        else:
            moves = likely_moves(pokemon, limit=8)
            if moves:
                for index, move in enumerate(moves):
                    weights[_key(move)] = 1.0 / (index + 1)

        for move in observed:
            weights[move] = max(weights.get(move, 0.0), 0.35)
        total = sum(weights.values())
        if total <= 0:
            return {}
        return {move: value / total for move, value in weights.items()}

    @staticmethod
    def _move_class(move: str) -> str:
        move = _key(move)
        if move in {"protect", "detect", "endure"}:
            return "protect"
        if move in {
            "swordsdance", "dragondance", "calmmind", "curse", "agility",
            "rockpolish", "bellydrum", "growth", "amnesia", "irondefense",
        }:
            return "setup"
        if move in {"toxic", "thunderwave", "willowisp", "leechseed", "roar", "whirlwind", "haze", "spikes", "rapidspin"}:
            return "passive"
        return "attack"

    def predict_responses(self, battle: Any, *, max_switches: int = 3, max_moves: int = 4) -> list[PredictedResponse]:
        self.observe_transition(battle)
        opponent = getattr(battle, "opponent_active_pokemon", None)
        if opponent is None:
            return []
        battle_id = str(getattr(battle, "battle_tag", "unknown"))
        behavior = self._behavior[battle_id]
        move_prior = self._profile_move_prior(opponent)
        responses: list[PredictedResponse] = []

        # Population prior: switches are conditional on the current species
        # and our active matchup. A weak/negative matchup increases the prior,
        # but behavioral evidence receives only a bounded Bayesian update.
        switch_prior = 0.18
        our_active = getattr(battle, "active_pokemon", None)
        if our_active is not None:
            our_types = {_key(t) for t in (getattr(our_active, "types", ()) or ())}
            opp_types = {_key(t) for t in (getattr(opponent, "types", ()) or ())}
            if our_types & {"electric", "ice", "fighting", "ground", "psychic", "fire", "water"}:
                if _base_stats(our_active).get("spa", 0) + _base_stats(our_active).get("atk", 0) > _base_stats(opponent).get("def", 0) + _base_stats(opponent).get("spd", 0):
                    switch_prior += 0.08
            if float(getattr(opponent, "current_hp_fraction", 1.0) or 1.0) < 0.30:
                switch_prior += 0.10
        behavioral_switch = behavior.smoothed_switch_rate()
        switch_prob = (switch_prior * self.prior_strength + behavioral_switch * max(1, behavior.switch_opportunities)) / (self.prior_strength + max(1, behavior.switch_opportunities))
        switch_prob = min(0.78, max(0.08, switch_prob))

        teammate_pool = []
        opponent_species = {_species(p) for p in (getattr(battle, "opponent_team", {}) or {}).values() if p is not None and not getattr(p, "fainted", False)}
        for p in (getattr(battle, "opponent_team", {}) or {}).values():
            if p is None or getattr(p, "fainted", False) or p is opponent:
                continue
            name = _species(p)
            if name in opponent_species:
                teammate_pool.append(p)
        scores = []
        our_types = tuple(_key(t) for t in (getattr(our_active, "types", ()) or ())) if our_active is not None else ()
        super_effective = {"fire": {"grass", "ice", "bug", "steel"}, "water": {"fire", "ground", "rock"}, "electric": {"water", "flying"}, "ice": {"grass", "ground", "flying", "dragon"}, "fighting": {"normal", "ice", "rock", "dark", "steel"}, "ground": {"fire", "electric", "poison", "rock", "steel"}, "psychic": {"fighting", "poison"}, "rock": {"fire", "ice", "flying", "bug"}}
        for p in teammate_pool:
            score = 0.0
            p_types = {_key(t) for t in (getattr(p, "types", ()) or ())}
            # Candidate is attractive when it resists at least one revealed
            # move type or is faster/bulkier than the current active.
            for move_type in move_prior:
                if move_type in super_effective and p_types & super_effective[move_type]:
                    score += 0.1
            score += 0.5 * float(getattr(p, "current_hp_fraction", 1.0) or 1.0)
            score += 0.002 * max(_base_stats(p).get("hp", 0), 0)
            scores.append((score, p))
        scores.sort(reverse=True, key=lambda x: x[0])
        top_switches = scores[:max_switches]
        if top_switches and switch_prob > 0:
            denom = sum(max(0.001, score) for score, _ in top_switches)
            for score, p in top_switches:
                probability = switch_prob * max(0.001, score) / denom
                responses.append(PredictedResponse("switch", probability, target=_species(p), rationale="population switch prior + smoothed in-battle tendency"))

        remaining_prob = max(0.0, 1.0 - switch_prob)
        attack_candidates = []
        for move, prob in sorted(move_prior.items(), key=lambda kv: kv[1], reverse=True):
            attack_candidates.append((move, prob))
        attack_candidates = attack_candidates[:max_moves]
        if attack_candidates:
            total = sum(prob for _, prob in attack_candidates)
            attack_mass = 0.0
            class_mass = defaultdict(float)
            for move, prob in attack_candidates:
                class_mass[self._move_class(move)] += prob / max(total, 1e-9)
            behavior_attack = behavior.smoothed_attack_rate()
            for move, prob in attack_candidates:
                move_share = prob / max(total, 1e-9)
                adjusted = remaining_prob * (0.65 * move_share + 0.35 * behavior_attack * move_share)
                responses.append(PredictedResponse(self._move_class(move), adjusted, move=move, rationale="Smogon set prior + revealed moves + Bayesian behavior smoothing"))

        # Ensure predictions approximately sum to one without fabricating a
        # specific hidden move. Residual uncertainty is represented explicitly.
        total = sum(max(0.0, r.probability) for r in responses)
        if total < 0.999:
            responses.append(PredictedResponse("unknown", 1.0 - total, rationale="unresolved hidden-information mass"))
        elif total > 1.001:
            responses = [PredictedResponse(r.kind, r.probability / total, r.target, r.move, r.rationale) for r in responses]
        return responses
