from __future__ import annotations

import copy
from typing import Any

from .damage import calculate_damage, type_multiplier
from metamon.interface import consistent_move_order


_SETUP_BOOSTS: dict[str, dict[str, int]] = {
    "dragondance": {"atk": 1, "spe": 1},
    "calmmind": {"spa": 1, "spd": 1},
    "swordsdance": {"atk": 2},
    "agility": {"spe": 2},
    "rockpolish": {"spe": 2},
    "curse": {"atk": 1, "def": 1, "spe": -1},
    "bellydrum": {"atk": 6},
}

_SETUP_MOVES = set(_SETUP_BOOSTS)
_PHazing = {"roar", "whirlwind"}
_GROUNDED_IMMUNE_TYPES = {"flying"}


def _move_id(move: Any) -> str:
    return str(getattr(move, "id", getattr(move, "name", ""))).lower().replace(" ", "")


def _base_power(move: Any) -> int:
    try:
        return int(getattr(move, "base_power", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _species_key(pokemon: Any) -> str:
    value = getattr(pokemon, "species", getattr(pokemon, "name", ""))
    return str(value).lower().replace(" ", "").replace("-", "")


def _hp_fraction(pokemon: Any) -> float:
    return float(getattr(pokemon, "current_hp_fraction", 0.0) or 0.0)


def _weather(battle: Any) -> str:
    return ",".join(
        str(getattr(x, "name", x)).lower()
        for x in (getattr(battle, "weather", {}) or {}).keys()
    )


def _ordered_moves(active: Any) -> list[Any]:
    moves = list((getattr(active, "moves", {}) or {}).values())
    try:
        return consistent_move_order(moves)
    except ValueError:
        return sorted(moves, key=_move_id)


def _move_index(active: Any, move: Any) -> int | None:
    for idx, candidate in enumerate(_ordered_moves(active)[:4]):
        if candidate is move or _move_id(candidate) == _move_id(move):
            return idx
    return None


def _revealed_opponents(battle: Any) -> list[Any]:
    return [
        p
        for p in (getattr(battle, "opponent_team", {}) or {}).values()
        if not getattr(p, "fainted", False)
    ]


def _our_alive_team(battle: Any) -> list[Any]:
    return [
        p
        for p in (getattr(battle, "team", {}) or {}).values()
        if not getattr(p, "fainted", False)
    ]


def _is_grounded(pokemon: Any) -> bool:
    types = {
        str(getattr(t, "name", t)).lower()
        for t in (getattr(pokemon, "types", ()) or ())
    }
    if types & _GROUNDED_IMMUNE_TYPES:
        return False
    ability = str(getattr(pokemon, "ability", "") or "").lower().replace(" ", "")
    if ability == "levitate":
        return False
    return True


def _has_revealed_move(pokemon: Any, ids: set[str]) -> bool:
    return any(_move_id(m) in ids for m in (getattr(pokemon, "moves", {}) or {}).values())


def _incoming_safe(battle: Any, active: Any, opponent: Any) -> tuple[bool, float | None, str]:
    """Require a real survival margin before recommending setup/hazard turns."""
    moves = [m for m in (getattr(opponent, "moves", {}) or {}).values() if _base_power(m) > 0]
    if not moves:
        return True, None, "no revealed damaging move"
    weather = _weather(battle)
    reliable = []
    for move in moves:
        result = calculate_damage(opponent, active, move, weather=weather)
        if result.reliable:
            reliable.append((result.percentage_max, _move_id(move)))
    if not reliable:
        return False, None, "revealed damage is not reliable enough for a free setup turn"
    worst_pct, worst_move = max(reliable, key=lambda x: x[0])
    current_hp_pct = _hp_fraction(active) * 100.0
    # Keep a material margin for rolls, status, and the following turn; this is
    # intentionally stricter than merely surviving the worst estimate.
    safe = worst_pct < max(20.0, current_hp_pct * 0.65)
    return safe, worst_pct, worst_move


def _boosted_clone(active: Any, boosts: dict[str, int]) -> Any | None:
    try:
        clone = copy.copy(active)
        current = dict(getattr(active, "boosts", {}) or {})
        for stat, delta in boosts.items():
            current[stat] = max(-6, min(6, int(current.get(stat, 0) or 0) + delta))
        clone.boosts = current
        return clone
    except Exception:
        return None


def _setup_candidate_value(battle: Any, active: Any, opponent: Any, move: Any) -> tuple[float, str]:
    """Score whether one setup turn materially converts the position."""
    move_id = _move_id(move)
    boosts = _SETUP_BOOSTS.get(move_id)
    if boosts is None:
        return 0.0, ""

    # Do not spend a setup turn when the opponent has revealed phazing unless
    # the resulting boosted move immediately ends the current threat.
    opponent_ids = {_move_id(m) for m in (getattr(opponent, "moves", {}) or {}).values()}
    has_phazing = bool(opponent_ids & _PHazing)

    weather = _weather(battle)
    current_ko = 0.0
    if opponent is not None:
        best_current = []
        for candidate in [m for m in _ordered_moves(active)[:4] if _base_power(m) > 0]:
            result = calculate_damage(active, opponent, candidate, weather=weather)
            if result.reliable:
                best_current.append(result.ko_probability)
        current_ko = max(best_current or [0.0])

    clone = _boosted_clone(active, boosts)
    if clone is None:
        return 0.0, "could not simulate boosted position"

    revealed = _revealed_opponents(battle)
    boosted_kos = 0
    meaningful_gain = 0.0
    for foe in revealed:
        attacks = [m for m in _ordered_moves(clone)[:4] if _base_power(m) > 0]
        best_now = 0.0
        best_after = 0.0
        for attack in attacks:
            before = calculate_damage(active, foe, attack, weather=weather)
            after = calculate_damage(clone, foe, attack, weather=weather)
            if before.reliable:
                best_now = max(best_now, before.ko_probability)
            if after.reliable:
                best_after = max(best_after, after.ko_probability)
        if best_after >= 1.0:
            boosted_kos += 1
        meaningful_gain += max(0.0, best_after - best_now)

    current_after = []
    for attack in [m for m in _ordered_moves(clone)[:4] if _base_power(m) > 0]:
        result = calculate_damage(clone, opponent, attack, weather=weather)
        if result.reliable:
            current_after.append(result.ko_probability)
    boosted_current_ko = max(current_after or [0.0])

    # Prefer setup that creates an actual sweep window, not setup for its own sake.
    if boosted_kos >= 2 and not has_phazing:
        return 10.0 + min(4.0, meaningful_gain * 4.0), (
            f"win-condition: {move_id} creates a boosted sweep window against {boosted_kos} revealed opponent(s)"
        )

    # One revealed target can still justify setup when the boost turns the
    # current interaction into a guaranteed KO and the position is otherwise safe.
    if boosted_current_ko >= 1.0 and current_ko < 1.0 and not has_phazing:
        return 6.0 + meaningful_gain, (
            f"win-condition: {move_id} changes the current matchup into a guaranteed KO"
        )

    return 0.0, ""


def strategic_opportunity_override(
    battle: Any,
    legal_actions: list[int],
    model_action: int,
) -> tuple[int | None, str]:
    """Find high-confidence Gen 3 win-condition opportunities.

    This layer is deliberately narrower than a full planner. It does not try
    to invent an opponent's unrevealed six; it acts only when the currently
    visible position supplies enough evidence to justify a strategic turn.
    """
    active = getattr(battle, "active_pokemon", None)
    opponent = getattr(battle, "opponent_active_pokemon", None)
    if active is None or opponent is None or getattr(battle, "force_switch", False):
        return None, ""

    moves = _ordered_moves(active)[:4]
    weather = _weather(battle)
    available_ids = {
        _move_id(m) for m in (getattr(battle, "available_moves", []) or [])
    }
    legal_moves = {
        int(a): moves[int(a)]
        for a in legal_actions
        if 0 <= int(a) < 4 and int(a) < len(moves)
        and _move_id(moves[int(a)]) in available_ids
    }

    # Never force setup/hazard play from a deeply compromised active.
    if _hp_fraction(active) < 0.50:
        return None, ""

    safe_turn, incoming_pct, incoming_note = _incoming_safe(battle, active, opponent)
    if not safe_turn:
        return None, ""

    opponent_ids = {_move_id(m) for m in (getattr(opponent, "moves", {}) or {}).values()}

    # --- Spikes stacking -------------------------------------------------
    spikes = 0
    for key, value in (getattr(battle, "opponent_side_conditions", {}) or {}).items():
        key_id = str(getattr(key, "name", key)).lower().replace(" ", "")
        if key_id == "spikes":
            try:
                spikes = int(value)
            except (TypeError, ValueError):
                spikes = 0
            break

    spikes_idx = next((idx for idx, move in legal_moves.items() if _move_id(move) == "spikes"), None)
    if spikes_idx is not None and spikes < 3:
        grounded_remaining = [p for p in _revealed_opponents(battle) if _is_grounded(p)]
        current_damage = []
        for attack in [m for m in moves if _base_power(m) > 0]:
            result = calculate_damage(active, opponent, attack, weather=weather)
            if result.reliable:
                current_damage.append(result.ko_probability)
        best_ko = max(current_damage or [0.0])
        has_spinner = "rapidspin" in opponent_ids or any(_has_revealed_move(p, {"rapidspin"}) for p in _revealed_opponents(battle))
        our_ghost = any(
            "ghost" in {str(getattr(t, "name", t)).lower() for t in (getattr(p, "types", ()) or ())}
            for p in _our_alive_team(battle)
        )

        # Stack only when the current active can actually buy the turn. Take an
        # immediate guaranteed KO instead of laying a gratuitous layer.
        if len(grounded_remaining) >= 2 and best_ko < 1.0:
            reason = (
                f"hazard plan: add Spikes layer {spikes + 1}/3; "
                f"{len(grounded_remaining)} grounded opponent(s) remain"
            )
            if spikes == 1 and has_spinner and not our_ghost:
                return None, ""
            if has_spinner and our_ghost:
                reason += "; preserve a spinblocker for the stack"
            if incoming_pct is not None:
                reason += f"; worst revealed damage {incoming_pct:.1f}% ({incoming_note})"
            return spikes_idx, reason

    # --- Setup / sweep window -------------------------------------------
    best: tuple[float, int, str] | None = None
    for action, move in legal_moves.items():
        value, reason = _setup_candidate_value(battle, active, opponent, move)
        if value <= 0:
            continue
        if best is None or value > best[0]:
            best = (value, action, reason)

    if best is not None:
        value, action, reason = best
        # Do not turn a guaranteed current KO into a setup turn.
        if 0 <= action < 4:
            selected = legal_moves[action]
            selected_id = _move_id(selected)
            if selected_id in _SETUP_MOVES:
                best_attacking_ko = 0.0
                for attack in moves:
                    if _base_power(attack) <= 0:
                        continue
                    result = calculate_damage(active, opponent, attack, weather=weather)
                    if result.reliable:
                        best_attacking_ko = max(best_attacking_ko, result.ko_probability)
                if best_attacking_ko >= 1.0:
                    return None, ""
        return action, reason + (f"; worst revealed incoming damage {incoming_pct:.1f}%" if incoming_pct is not None else "")

    return None, ""
