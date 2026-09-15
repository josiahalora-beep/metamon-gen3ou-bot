from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

from .damage import calculate_damage, type_multiplier
from metamon.interface import consistent_move_order, consistent_pokemon_order

_STATUS_HAZARD = {"tox", "psn", "brn", "poison", "burn"}
_RECOVERY_OR_STALL = {"recover", "softboiled", "rest", "protect", "leechseed", "toxic", "spikes", "roar", "whirlwind"}


def _move_type(move: Any) -> str:
    value = getattr(move, "type", "")
    return str(getattr(value, "name", value)).lower()


def _move_id(move: Any) -> str:
    return str(getattr(move, "id", getattr(move, "name", ""))).lower().replace(" ", "")


def _base_power(move: Any) -> int:
    return int(getattr(move, "base_power", 0) or 0)


def _revealed_moves(pokemon: Any) -> list[Any]:
    if pokemon is None:
        return []
    return list((getattr(pokemon, "moves", {}) or {}).values())


def _revealed_damaging_moves(pokemon: Any) -> list[Any]:
    return [m for m in _revealed_moves(pokemon) if _base_power(m) > 0]


def _fallback_pokemon_key(pokemon: Any) -> str:
    value = getattr(pokemon, "species", getattr(pokemon, "name", ""))
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _switch_slots(battle: Any) -> list[Any]:
    """Return switch slots using exactly the same ordering contract as evaluator."""
    team = [
        p for p in (getattr(battle, "team", {}) or {}).values()
        if not getattr(p, "fainted", False) and not getattr(p, "active", False)
    ]
    if not team:
        return []
    try:
        return consistent_pokemon_order(team)
    except ValueError:
        # Unit-test doubles may not be poke-env Pokemon objects. Mirror the
        # evaluator's fallback ordering rather than returning no target.
        return sorted(team, key=_fallback_pokemon_key)


def _team_pokemon_for_action(battle: Any, action: int) -> Any | None:
    """Resolve action 4-8 to the exact canonical switch target."""
    if action < 4:
        return None
    slots = _switch_slots(battle)
    switch_index = action - 4
    return slots[switch_index] if 0 <= switch_index < len(slots) else None


def _current_hp_fraction(pokemon: Any) -> float:
    return float(getattr(pokemon, "current_hp_fraction", 0.0) or 0.0)


def _status_name(pokemon: Any) -> str:
    value = getattr(pokemon, "status", "")
    return str(getattr(value, "name", value)).lower()


def incoming_damage_profile(opponent: Any, defender: Any, *, weather: str = "") -> dict:
    """Estimate the worst revealed attack against a candidate switch."""
    moves = _revealed_damaging_moves(opponent)
    if defender is None or not moves:
        return {"known_moves": 0, "max_damage_pct": None, "guaranteed_ko": False, "moves": []}
    results = []
    for move in moves:
        result = calculate_damage(opponent, defender, move, weather=weather)
        if result.reliable:
            results.append((result.percentage_max, result.ko_probability, _move_id(move)))
    if not results:
        return {"known_moves": len(moves), "max_damage_pct": None, "guaranteed_ko": False, "moves": [_move_id(m) for m in moves]}
    return {
        "known_moves": len(moves),
        "max_damage_pct": max(r[0] for r in results),
        "guaranteed_ko": any(r[1] >= 1.0 for r in results),
        "moves": [_move_id(m) for m in moves],
    }


def incoming_type_profile(opponent: Any, defender: Any) -> dict:
    moves = _revealed_damaging_moves(opponent)
    if defender is None or not moves:
        return {"known_moves": 0, "max_multiplier": None, "min_multiplier": None, "super_effective": False}
    defender_types = list(getattr(defender, "types", ()) or ())
    multipliers = [type_multiplier(_move_type(move), defender_types) for move in moves]
    return {
        "known_moves": len(moves),
        "max_multiplier": max(multipliers),
        "min_multiplier": min(multipliers),
        "super_effective": any(m >= 2.0 for m in multipliers),
        "immune": any(m == 0.0 for m in multipliers),
        "moves": [_move_id(m) for m in moves],
    }


def _weather(battle: Any) -> str:
    return ",".join(str(getattr(x, "name", x)).lower() for x in (getattr(battle, "weather", {}) or {}).keys())


def _dangerous_pokemon(pokemon: Any, opponent: Any, *, battle: Any = None) -> bool:
    if pokemon is None or getattr(pokemon, "fainted", False):
        return False
    hp = _current_hp_fraction(pokemon)
    status = _status_name(pokemon)
    profile = incoming_type_profile(opponent, pokemon)
    damage = incoming_damage_profile(opponent, pokemon, weather=_weather(battle) if battle else "")
    if not profile["known_moves"]:
        return hp <= 0.15
    if hp <= 0.15:
        return True
    if hp <= 0.30 and status in _STATUS_HAZARD:
        return True
    if hp <= 0.25 and profile["super_effective"]:
        return True
    if damage["guaranteed_ko"]:
        return True
    return False


def _switch_safety_score(pokemon: Any, opponent: Any, *, battle: Any = None) -> tuple[float, dict]:
    profile = incoming_type_profile(opponent, pokemon)
    damage = incoming_damage_profile(opponent, pokemon, weather=_weather(battle) if battle else "")
    hp = _current_hp_fraction(pokemon)
    if not profile["known_moves"]:
        return (hp, {**profile, **damage})
    max_mult = float(profile["max_multiplier"] or 1.0)
    status = _status_name(pokemon)
    score = 2.0 * hp - 2.5 * max_mult
    if status in _STATUS_HAZARD:
        score -= 0.5
    if max_mult == 0.0:
        score += 2.0
    elif max_mult < 1.0:
        score += 1.0
    if damage["guaranteed_ko"]:
        score -= 5.0
    elif damage["max_damage_pct"] is not None:
        score -= min(4.0, float(damage["max_damage_pct"]) / 100.0 * 3.0)
    return score, {**profile, **damage}


@dataclass(frozen=True)
class SafetyDecision:
    action: int | None
    reason: str = ""
    hard: bool = False


def _ordered_moves(active: Any) -> list[Any]:
    moves = _revealed_moves(active)
    try:
        return consistent_move_order(moves)
    except ValueError:
        return sorted(moves, key=_move_id)


def _selected_move(active: Any, action: int) -> Any | None:
    if action < 0 or action >= 4:
        return None
    moves = _ordered_moves(active)
    return moves[action] if action < len(moves) else None


def safety_override(battle: Any, legal_actions: list[int], model_action: int) -> SafetyDecision:
    """Return high-confidence anti-throw corrections while preserving healthy value."""
    active = getattr(battle, "active_pokemon", None)
    opponent = getattr(battle, "opponent_active_pokemon", None)
    if active is None or opponent is None:
        return SafetyDecision(None)

    # A sub-35% active is at risk of being lost before it can be repositioned.
    # Keep it alive unless the selected move is a demonstrated guaranteed KO.
    if 0 <= model_action < 4 and _current_hp_fraction(active) <= 0.35:
        selected = _selected_move(active, model_action)
        guaranteed_ko = False
        if selected is not None and _base_power(selected) > 0:
            result = calculate_damage(active, opponent, selected, weather=_weather(battle))
            guaranteed_ko = result.reliable and result.ko_probability >= 1.0
        if not guaranteed_ko:
            alternatives = []
            for action in legal_actions:
                if action < 4:
                    continue
                candidate = _team_pokemon_for_action(battle, action)
                if candidate is None or getattr(candidate, "fainted", False):
                    continue
                score, profile = _switch_safety_score(candidate, opponent, battle=battle)
                if profile.get("guaranteed_ko"):
                    continue
                alternatives.append((score, int(action), profile, candidate))
            if alternatives:
                best_score, best_action, best_profile, best_target = max(alternatives, key=lambda x: (x[0], -x[1]))
                return SafetyDecision(
                    best_action,
                    reason=(f"low-HP preservation: refused non-guaranteed attack with active "
                            f"{getattr(active, 'species', 'pokemon')} at {_current_hp_fraction(active):.0%}; "
                            f"switch to {getattr(best_target, 'species', 'pokemon')} "
                            f"(HP={_current_hp_fraction(best_target):.0%})"),
                    hard=True,
                )

    if model_action >= 4 and model_action in legal_actions:
        chosen_target = _team_pokemon_for_action(battle, model_action)
        if _dangerous_pokemon(chosen_target, opponent, battle=battle):
            alternatives = []
            for action in legal_actions:
                if action < 4 or action == model_action:
                    continue
                candidate = _team_pokemon_for_action(battle, action)
                if candidate is None:
                    continue
                score, profile = _switch_safety_score(candidate, opponent, battle=battle)
                alternatives.append((score, int(action), profile))
            if alternatives:
                current_score, current_profile = _switch_safety_score(chosen_target, opponent, battle=battle)
                best_score, best_action, best_profile = max(alternatives, key=lambda item: (item[0], -item[1]))
                if best_score > current_score + 1.0:
                    return SafetyDecision(
                        best_action,
                        reason=(f"anti-throw: refused switch into compromised {getattr(chosen_target, 'species', 'pokemon')} "
                                f"(HP={_current_hp_fraction(chosen_target):.0%}); safer switch has estimated worst revealed damage "
                                f"{best_profile.get('max_damage_pct')}% vs {current_profile.get('max_damage_pct')}%"),
                        hard=True,
                    )

    if model_action < 4 and _dangerous_pokemon(active, opponent, battle=battle):
        alternatives = []
        for action in legal_actions:
            if action < 4:
                continue
            candidate = _team_pokemon_for_action(battle, action)
            if candidate is None:
                continue
            score, profile = _switch_safety_score(candidate, opponent, battle=battle)
            alternatives.append((score, int(action), profile))
        if alternatives:
            best_score, best_action, best_profile = max(alternatives, key=lambda item: (item[0], -item[1]))
            active_score, active_profile = _switch_safety_score(active, opponent, battle=battle)
            if best_score > active_score + 0.75:
                return SafetyDecision(
                    best_action,
                    reason=(f"anti-throw: active {getattr(active, 'species', 'pokemon')} is compromised "
                            f"(HP={_current_hp_fraction(active):.0%}); safer switch estimated at "
                            f"{best_profile.get('max_damage_pct')}% worst revealed damage vs "
                            f"{active_profile.get('max_damage_pct')}%"),
                    hard=True,
                )

    boosts = getattr(opponent, "boosts", {}) or {}
    offensive_boost = max(int(boosts.get("atk", 0) or 0), int(boosts.get("spa", 0) or 0), int(boosts.get("spe", 0) or 0))
    if offensive_boost >= 2 and model_action < 4:
        selected = _selected_move(active, model_action)
        if selected is None or _base_power(selected) <= 0:
            candidates = []
            for idx, move in enumerate(_ordered_moves(active)[:4]):
                if _base_power(move) <= 0:
                    continue
                result = calculate_damage(active, opponent, move, weather=_weather(battle))
                if result.reliable:
                    candidates.append((result.max_damage, idx, move, result))
            if candidates:
                _, best_idx, best_move, result = max(candidates, key=lambda x: x[0])
                return SafetyDecision(best_idx, reason=(f"setup emergency: opponent has +{offensive_boost} offensive boost; "
                                f"use {getattr(best_move, 'name', getattr(best_move, 'id', 'attack'))} "
                                f"({result.percentage_max:.0f}% estimated max damage)"), hard=True)

    opponent_hp = _current_hp_fraction(opponent)
    revealed_ids = {_move_id(m) for m in _revealed_moves(opponent)}
    has_stall_tools = bool(revealed_ids & _RECOVERY_OR_STALL)
    if model_action < 4 and opponent_hp <= 0.30 and has_stall_tools:
        selected = _selected_move(active, model_action)
        if selected is None or _base_power(selected) <= 0:
            candidates = []
            for idx, move in enumerate(_ordered_moves(active)[:4]):
                if _base_power(move) <= 0:
                    continue
                result = calculate_damage(active, opponent, move, weather=_weather(battle))
                if result.reliable and result.ko_probability >= 1.0:
                    candidates.append((result.max_damage, idx, move))
            if candidates:
                _, idx, move = max(candidates, key=lambda x: x[0])
                return SafetyDecision(idx, reason=(f"stall conversion: revealed recovery/hazard tools {sorted(revealed_ids & _RECOVERY_OR_STALL)} "
                                f"and opponent is at {opponent_hp:.0%}; take guaranteed KO"), hard=True)

    return SafetyDecision(None)
