from __future__ import annotations

from typing import Any

from .damage import calculate_damage
from metamon.interface import consistent_move_order, consistent_pokemon_order


def _moves(pokemon: Any) -> list[Any]:
    raw = list((getattr(pokemon, "moves", {}) or {}).values()) if pokemon is not None else []
    try:
        return list(consistent_move_order(raw))
    except ValueError:
        return sorted(raw, key=lambda m: str(getattr(m, "id", getattr(m, "name", ""))))


def _switches(battle: Any) -> list[Any]:
    raw = [p for p in (getattr(battle, "team", {}) or {}).values()
           if p is not None and not getattr(p, "fainted", False) and not getattr(p, "active", False)]
    try:
        return list(consistent_pokemon_order(raw))
    except ValueError:
        return sorted(raw, key=lambda p: str(getattr(p, "species", getattr(p, "name", ""))))


def _weather(battle: Any) -> str:
    return ",".join(str(getattr(k, "name", k)).lower() for k in (getattr(battle, "weather", {}) or {}).keys())


def hard_loss_switch_allowed(battle: Any, model_action: int, switch_action: int) -> bool:
    """Permit a switch override only for a demonstrated immediate loss."""
    if model_action >= 4 or switch_action < 4:
        return False
    active = getattr(battle, "active_pokemon", None)
    opponent = getattr(battle, "opponent_active_pokemon", None)
    if active is None or opponent is None or getattr(active, "fainted", False):
        return False

    weather = _weather(battle)
    moves = _moves(active)
    selected = moves[model_action] if 0 <= model_action < len(moves) else None
    if selected is None:
        return False

    # A hard-loss gate is intentionally narrow. If the learned attack can
    # meaningfully punish the current opponent, preserve the model action even
    # when the attacker is fragile. Small test doubles may expose only the KO
    # probability, so percentage_max is treated as an optional field.
    if int(getattr(selected, "base_power", 0) or 0) > 0:
        result = calculate_damage(active, opponent, selected, weather=weather)
        if result.reliable:
            if result.ko_probability >= 0.50:
                return False
            percentage_max = getattr(result, "percentage_max", None)
            if percentage_max is not None and percentage_max > 20.0:
                return False

    incoming = []
    for move in (getattr(opponent, "moves", {}) or {}).values():
        if int(getattr(move, "base_power", 0) or 0) <= 0:
            continue
        result = calculate_damage(opponent, active, move, weather=weather)
        if result.reliable:
            incoming.append(result)
    if not any(r.ko_probability >= 1.0 for r in incoming):
        return False

    slots = _switches(battle)
    idx = switch_action - 4
    if not (0 <= idx < len(slots)):
        return False
    candidate = slots[idx]
    candidate_results = []
    for move in (getattr(opponent, "moves", {}) or {}).values():
        if int(getattr(move, "base_power", 0) or 0) <= 0:
            continue
        result = calculate_damage(opponent, candidate, move, weather=weather)
        if result.reliable:
            candidate_results.append(result)
    return not any(r.ko_probability >= 1.0 for r in candidate_results)
