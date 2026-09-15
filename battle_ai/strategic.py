from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .damage import calculate_damage, type_multiplier


_STATUS_HAZARD = {"tox", "psn", "brn", "poison", "burn"}
_NON_DAMAGING_CATEGORIES = {"status", ""}


def _move_type(move: Any) -> str:
    value = getattr(move, "type", "")
    return str(getattr(value, "name", value)).lower()


def _move_id(move: Any) -> str:
    return str(getattr(move, "id", getattr(move, "name", ""))).lower().replace(" ", "")


def _base_power(move: Any) -> int:
    return int(getattr(move, "base_power", 0) or 0)


def _revealed_damaging_moves(pokemon: Any) -> list[Any]:
    if pokemon is None:
        return []
    moves = list((getattr(pokemon, "moves", {}) or {}).values())
    return [m for m in moves if _base_power(m) > 0]


def _team_pokemon_for_action(battle: Any, action: int) -> Any | None:
    if action < 4:
        return None
    switch_index = action - 4
    team = [
        p for p in (getattr(battle, "team", {}) or {}).values()
        if not getattr(p, "fainted", False) and not getattr(p, "active", False)
    ]
    team.sort(key=lambda p: str(getattr(p, "name", getattr(p, "species", ""))))
    return team[switch_index] if 0 <= switch_index < len(team) else None


def _current_hp_fraction(pokemon: Any) -> float:
    return float(getattr(pokemon, "current_hp_fraction", 0.0) or 0.0)


def _status_name(pokemon: Any) -> str:
    value = getattr(pokemon, "status", "")
    return str(getattr(value, "name", value)).lower()


def incoming_type_profile(opponent: Any, defender: Any) -> dict:
    """Summarize type-level exposure to revealed opponent attacks.

    This intentionally ignores unrevealed moves and unknown damage statistics.
    It is a risk signal, not a damage prediction.
    """
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
        "moves": [_move_id(move) for move in moves],
    }


def _dangerous_pokemon(pokemon: Any, opponent: Any) -> bool:
    if pokemon is None or getattr(pokemon, "fainted", False):
        return False
    hp = _current_hp_fraction(pokemon)
    status = _status_name(pokemon)
    profile = incoming_type_profile(opponent, pokemon)
    if not profile["known_moves"]:
        return False
    if hp <= 0.15:
        return True
    if hp <= 0.30 and status in _STATUS_HAZARD:
        return True
    if hp <= 0.25 and profile["super_effective"]:
        return True
    return False


def _switch_safety_score(pokemon: Any, opponent: Any) -> tuple[float, dict]:
    profile = incoming_type_profile(opponent, pokemon)
    if not profile["known_moves"]:
        return (0.0, profile)
    max_mult = float(profile["max_multiplier"] or 1.0)
    hp = _current_hp_fraction(pokemon)
    status = _status_name(pokemon)
    score = 2.0 * hp - 2.5 * max_mult
    if status in _STATUS_HAZARD:
        score -= 0.5
    if max_mult == 0.0:
        score += 2.0
    elif max_mult < 1.0:
        score += 1.0
    return score, profile


@dataclass(frozen=True)
class SafetyDecision:
    action: int | None
    reason: str = ""
    hard: bool = False


def safety_override(battle: Any, legal_actions: list[int], model_action: int) -> SafetyDecision:
    """Return only high-confidence anti-throw corrections.

    The safety layer deliberately does not invent opponent stats or sets. It
    only reacts to information already present in poke-env (revealed moves,
    known HP/status, and known boosts) and otherwise leaves the model alone.
    """
    active = getattr(battle, "active_pokemon", None)
    opponent = getattr(battle, "opponent_active_pokemon", None)
    if active is None or opponent is None:
        return SafetyDecision(None)

    # 1) Never switch a critically compromised team member into a known-danger
    # position when another legal switch is materially safer.
    if model_action >= 4 and model_action in legal_actions:
        chosen_target = _team_pokemon_for_action(battle, model_action)
        if _dangerous_pokemon(chosen_target, opponent):
            alternatives: list[tuple[float, int, dict]] = []
            for action in legal_actions:
                if action < 4 or action == model_action:
                    continue
                candidate = _team_pokemon_for_action(battle, action)
                if candidate is None:
                    continue
                score, profile = _switch_safety_score(candidate, opponent)
                alternatives.append((score, int(action), profile))
            if alternatives:
                current_score, _, current_profile = _switch_safety_score(chosen_target, opponent)
                best_score, best_action, best_profile = max(alternatives, key=lambda item: item[0])
                if best_score > current_score + 1.0:
                    return SafetyDecision(
                        best_action,
                        reason=(
                            f"anti-throw: refused switch into compromised {getattr(chosen_target, 'species', 'pokemon')} "
                            f"(HP={_current_hp_fraction(chosen_target):.0%}, status={_status_name(chosen_target) or 'healthy'}); "
                            f"safer switch available with known-move max type risk "
                            f"{best_profile.get('max_multiplier')}x vs {current_profile.get('max_multiplier')}x"
                        ),
                        hard=True,
                    )

    # 2) If the active Pokemon is critically compromised, don't spend the turn
    # on another move when a materially safer legal switch exists.
    if model_action < 4 and _dangerous_pokemon(active, opponent):
        profile = incoming_type_profile(opponent, active)
        alternatives: list[tuple[float, int, dict]] = []
        for action in legal_actions:
            if action < 4:
                continue
            candidate = _team_pokemon_for_action(battle, action)
            if candidate is None:
                continue
            score, candidate_profile = _switch_safety_score(candidate, opponent)
            alternatives.append((score, int(action), candidate_profile))
        if alternatives:
            best_score, best_action, best_profile = max(alternatives, key=lambda item: item[0])
            active_score, _ = _switch_safety_score(active, opponent)
            if best_score > active_score + 0.75:
                return SafetyDecision(
                    best_action,
                    reason=(
                        f"anti-throw: active {getattr(active, 'species', 'pokemon')} is compromised "
                        f"(HP={_current_hp_fraction(active):.0%}, status={_status_name(active) or 'healthy'}); "
                        f"safer switch has known-move max type risk {best_profile.get('max_multiplier')}x "
                        f"vs active {profile.get('max_multiplier')}x"
                    ),
                    hard=True,
                )

    # 3) Emergency response to a revealed setup sweeper. Only override a
    # non-damaging move when we can calculate a reliable damaging alternative.
    boosts = getattr(opponent, "boosts", {}) or {}
    offensive_boost = max(int(boosts.get("atk", 0) or 0), int(boosts.get("spa", 0) or 0), int(boosts.get("spe", 0) or 0))
    if offensive_boost >= 2 and model_action < 4:
        active_moves = list((getattr(active, "moves", {}) or {}).values())
        selected_move = None
        if 0 <= model_action < 4:
            ordered = sorted(active_moves, key=lambda m: _move_id(m))
            if model_action < len(ordered):
                selected_move = ordered[model_action]
        if selected_move is None or _base_power(selected_move) <= 0:
            candidates: list[tuple[int, int, Any]] = []
            for idx, move in enumerate(sorted(active_moves, key=lambda m: _move_id(m))[:4]):
                if _base_power(move) <= 0:
                    continue
                result = calculate_damage(
                    active,
                    opponent,
                    move,
                    weather="".join(str(x) for x in (getattr(battle, "weather", {}) or {}).keys()),
                )
                if result.reliable:
                    candidates.append((result.max_damage, idx, move))
            if candidates:
                _, best_idx, best_move = max(candidates, key=lambda x: x[0])
                return SafetyDecision(
                    best_idx,
                    reason=(
                        f"setup emergency: opponent has +{offensive_boost} offensive boost; "
                        f"replace non-damaging model action with {getattr(best_move, 'name', getattr(best_move, 'id', 'attack'))}"
                    ),
                    hard=True,
                )

    return SafetyDecision(None)
