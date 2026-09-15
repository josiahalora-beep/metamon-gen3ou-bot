from __future__ import annotations

from typing import Any

from .damage import calculate_damage
from metamon.interface import consistent_move_order, consistent_pokemon_order

SELF_KO_MOVES = {"explosion", "selfdestruct", "memento"}


def _move_id(move: Any) -> str:
    return str(getattr(move, "id", getattr(move, "name", "")) or "").lower().replace(" ", "").replace("-", "").replace("_", "")


def _power(move: Any) -> int:
    try:
        return int(getattr(move, "base_power", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _moves(active: Any) -> list[Any]:
    raw = list((getattr(active, "moves", {}) or {}).values()) if active is not None else []
    try:
        return list(consistent_move_order(raw))
    except ValueError:
        return sorted(raw, key=_move_id)


def _switches(battle: Any) -> list[Any]:
    raw = [p for p in (getattr(battle, "team", {}) or {}).values()
           if p is not None and not getattr(p, "fainted", False) and not getattr(p, "active", False)]
    try:
        return list(consistent_pokemon_order(raw))
    except ValueError:
        return sorted(raw, key=lambda p: str(getattr(p, "species", getattr(p, "name", ""))))


def _weather(battle: Any) -> str:
    return ",".join(str(getattr(x, "name", x)).lower() for x in (getattr(battle, "weather", {}) or {}).keys())


def _revealed_damaging_moves(pokemon: Any) -> list[Any]:
    if pokemon is None:
        return []
    return [m for m in (getattr(pokemon, "moves", {}) or {}).values() if _power(m) > 0]


def _incoming_profile(opponent: Any, candidate: Any, battle: Any) -> tuple[float, bool]:
    worst = 0.0
    guaranteed_ko = False
    for move in _revealed_damaging_moves(opponent):
        result = calculate_damage(opponent, candidate, move, weather=_weather(battle))
        if not result.reliable:
            continue
        worst = max(worst, float(getattr(result, "percentage_max", 0.0) or 0.0))
        if float(getattr(result, "ko_probability", 0.0) or 0.0) >= 1.0:
            guaranteed_ko = True
    return worst, guaranteed_ko


def _safe_switch_action(battle: Any, legal_actions: list[int]) -> int | None:
    opponent = getattr(battle, "opponent_active_pokemon", None)
    switches = _switches(battle)
    candidates: list[tuple[float, float, int]] = []
    for action in legal_actions:
        action = int(action)
        if action < 4:
            continue
        idx = action - 4
        if not (0 <= idx < len(switches)):
            continue
        candidate = switches[idx]
        worst, guaranteed = _incoming_profile(opponent, candidate, battle)
        if guaranteed:
            continue
        hp = float(getattr(candidate, "current_hp_fraction", 1.0) or 1.0)
        # Lower incoming damage is primary; preserving a healthy reserve is secondary.
        candidates.append((worst, -hp, action))
    if not candidates:
        return None
    return min(candidates)[2]


def immediate_loss_guard(battle: Any, legal_actions: list[int], chosen: int) -> tuple[int | None, str]:
    """Only intervene when the chosen action is mechanically losing right now.

    This guard deliberately ignores speculative matchup narratives. It exists to
    prevent the learned policy/search stack from selecting a non-KO attack into a
    revealed guaranteed KO, or selecting a non-winning self-KO move.
    """
    active = getattr(battle, "active_pokemon", None)
    opponent = getattr(battle, "opponent_active_pokemon", None)
    if active is None or opponent is None:
        return None, ""

    legal = {int(x) for x in legal_actions}
    if chosen not in legal:
        return None, ""

    incoming_guaranteed = False
    incoming_move = ""
    for move in _revealed_damaging_moves(opponent):
        result = calculate_damage(opponent, active, move, weather=_weather(battle))
        if result.reliable and float(getattr(result, "ko_probability", 0.0) or 0.0) >= 1.0:
            incoming_guaranteed = True
            incoming_move = _move_id(move)
            break

    if chosen < 4:
        move_slots = _moves(active)
        if not (0 <= chosen < len(move_slots)):
            return None, ""
        selected = move_slots[chosen]
        selected_result = calculate_damage(active, opponent, selected, weather=_weather(battle)) if _power(selected) > 0 else None
        selected_wins = bool(selected_result and selected_result.reliable and float(getattr(selected_result, "ko_probability", 0.0) or 0.0) >= 1.0)
        self_ko = _move_id(selected) in SELF_KO_MOVES

        if self_ko and not selected_wins:
            action = _safe_switch_action(battle, legal_actions)
            if action is not None:
                return action, f"hard intelligence guard: rejected non-winning { _move_id(selected) } self-KO"

        if incoming_guaranteed and not selected_wins:
            action = _safe_switch_action(battle, legal_actions)
            if action is not None:
                return action, (
                    f"hard intelligence guard: { _move_id(selected) } does not KO, while revealed "
                    f"{incoming_move} has a guaranteed KO; preserve the active"
                )

    else:
        switches = _switches(battle)
        idx = chosen - 4
        if 0 <= idx < len(switches):
            _, guaranteed = _incoming_profile(opponent, switches[idx], battle)
            if guaranteed:
                action = _safe_switch_action(battle, legal_actions)
                if action is not None and action != chosen:
                    return action, "hard intelligence guard: rejected a switch that is guaranteed to be KO'd"

    return None, ""
