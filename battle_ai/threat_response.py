from __future__ import annotations

from typing import Any

from metamon.interface import consistent_pokemon_order


# Gen 3 type chart compressed into only the relationships that matter for a
# conservative defensive switch. Unlisted pairs are neutral (1.0).
_SUPER_EFFECTIVE: dict[str, set[str]] = {
    "normal": set(),
    "fire": {"grass", "ice", "bug", "steel"},
    "water": {"fire", "ground", "rock"},
    "electric": {"water", "flying"},
    "grass": {"water", "ground", "rock"},
    "ice": {"grass", "ground", "flying", "dragon"},
    "fighting": {"normal", "ice", "rock", "dark", "steel"},
    "poison": {"grass"},
    "ground": {"fire", "electric", "poison", "rock", "steel"},
    "flying": {"grass", "fighting", "bug"},
    "psychic": {"fighting", "poison"},
    "bug": {"grass", "psychic", "dark"},
    "rock": {"fire", "ice", "flying", "bug"},
    "ghost": {"ghost", "psychic"},
    "dragon": {"dragon"},
    "dark": {"psychic", "ghost"},
    "steel": {"ice", "rock"},
}
_RESISTED: dict[str, set[str]] = {
    "normal": {"rock", "steel"},
    "fire": {"fire", "water", "rock", "dragon"},
    "water": {"water", "grass", "dragon"},
    "electric": {"electric", "grass", "dragon"},
    "grass": {"fire", "grass", "poison", "flying", "bug", "dragon", "steel"},
    "ice": {"fire", "water", "ice", "steel"},
    "fighting": {"poison", "flying", "psychic", "bug"},
    "poison": {"poison", "ground", "rock", "ghost"},
    "ground": {"grass", "bug"},
    "flying": {"electric", "rock", "steel"},
    "psychic": {"steel", "psychic"},
    "bug": {"fire", "fighting", "poison", "flying", "ghost", "steel"},
    "rock": {"fighting", "ground", "steel"},
    "ghost": {"dark"},
    "dragon": {"steel"},
    "dark": {"fighting", "dark", "steel"},
    "steel": {"fire", "water", "electric", "steel"},
}
_IMMUNE: dict[str, set[str]] = {
    "normal": {"ghost"},
    "fighting": {"ghost"},
    "electric": {"ground"},
    "poison": {"steel"},
    "ground": {"flying"},
    "psychic": {"dark"},
    "ghost": {"normal"},
    "dragon": set(),
}


def _key(value: Any) -> str:
    return str(getattr(value, "name", getattr(value, "species", value))).lower().replace(" ", "").replace("-", "")


def _types(pokemon: Any) -> tuple[str, ...]:
    return tuple(_key(t) for t in (getattr(pokemon, "types", ()) or ()) if t is not None)


def _base_stats(pokemon: Any) -> dict[str, int]:
    raw = getattr(pokemon, "base_stats", {}) or {}
    try:
        return {str(k).lower(): int(v) for k, v in raw.items() if v is not None}
    except (AttributeError, TypeError, ValueError):
        return {}


def _stat(pokemon: Any, stat: str) -> float:
    """Use observed battle stats when available, otherwise exact Gen 3 base stats."""
    stats = getattr(pokemon, "stats", {}) or {}
    try:
        value = stats.get(stat) if isinstance(stats, dict) else getattr(stats, stat, None)
        if value is not None:
            return float(value)
    except (TypeError, ValueError):
        pass
    return float(_base_stats(pokemon).get(stat, 0))


def _boosted_stat(pokemon: Any, stat: str) -> float:
    value = _stat(pokemon, stat)
    boosts = getattr(pokemon, "boosts", {}) or {}
    try:
        boost = int(boosts.get(stat, 0) or 0)
    except (TypeError, ValueError):
        boost = 0
    if boost >= 0:
        multiplier = (2.0 + boost) / 2.0
    else:
        multiplier = 2.0 / (2.0 - boost)
    return value * multiplier


def _type_multiplier(attack_type: str, defender: Any) -> float:
    multiplier = 1.0
    for defense_type in _types(defender):
        if defense_type in _IMMUNE.get(attack_type, set()):
            multiplier *= 0.0
        elif defense_type in _SUPER_EFFECTIVE.get(attack_type, set()):
            multiplier *= 2.0
        elif defense_type in _RESISTED.get(attack_type, set()):
            multiplier *= 0.5
    return multiplier


def _switch_slots(battle: Any) -> list[Any]:
    team = [
        p for p in (getattr(battle, "team", {}) or {}).values()
        if not getattr(p, "fainted", False) and not getattr(p, "active", False)
    ]
    if not team:
        return []
    try:
        return consistent_pokemon_order(team)
    except ValueError:
        return sorted(team, key=lambda p: _key(p))


def _passive(move: Any) -> bool:
    return int(getattr(move, "base_power", 0) or 0) <= 0


def hidden_threat_switch_override(
    battle: Any,
    legal_actions: list[int],
    model_action: int,
    move_map: dict[int, Any],
) -> tuple[int | None, str]:
    """Use known Gen 3 species typing/base stats without fabricating hidden moves.

    This deliberately requires several signals before overriding the model:
    a passive selected move, a faster high-offense opposing active, and a teammate
    with materially better defensive bulk or typing.
    """
    if not (0 <= int(model_action) < 4):
        return None, ""
    selected = move_map.get(int(model_action))
    if selected is None or not _passive(selected):
        return None, ""

    active = getattr(battle, "active_pokemon", None)
    opponent = getattr(battle, "opponent_active_pokemon", None)
    if active is None or opponent is None or getattr(opponent, "fainted", False):
        return None, ""

    active_speed = _boosted_stat(active, "spe")
    opponent_speed = _boosted_stat(opponent, "spe")
    opponent_atk = _boosted_stat(opponent, "atk")
    opponent_spa = _boosted_stat(opponent, "spa")
    special_threat = opponent_spa >= opponent_atk and opponent_spa >= 90
    physical_threat = opponent_atk > opponent_spa and opponent_atk >= 100
    speed_threat = opponent_speed >= active_speed * 1.10
    if not (special_threat or physical_threat) or not speed_threat:
        return None, ""

    use_special = special_threat and not physical_threat
    active_bulk_stat = "spd" if use_special else "def"
    active_bulk = _stat(active, "hp") + _boosted_stat(active, active_bulk_stat)

    opponent_type_pressure = _types(opponent)
    active_matchup = 1.0
    for attack_type in opponent_type_pressure:
        active_matchup = max(active_matchup, _type_multiplier(attack_type, active))
    if active_matchup < 1.0:
        return None, ""

    best: tuple[float, int, Any, float, float] | None = None
    slots = _switch_slots(battle)
    for action in legal_actions:
        if action < 4:
            continue
        idx = int(action) - 4
        if not (0 <= idx < len(slots)):
            continue
        candidate = slots[idx]
        if _stat(candidate, "hp") <= 0 or float(getattr(candidate, "current_hp_fraction", 0.0) or 0.0) < 0.50:
            continue

        matchup = 1.0
        for attack_type in opponent_type_pressure:
            matchup = min(matchup, _type_multiplier(attack_type, candidate))
        bulk = _stat(candidate, "hp") + _boosted_stat(candidate, active_bulk_stat)
        bulk_gain = bulk - active_bulk
        matchup_gain = active_matchup - matchup

        score = matchup_gain * 40.0 + max(0.0, bulk_gain) / 5.0
        if use_special and bulk_gain > 35:
            score += 8.0
        if matchup <= 0.5:
            score += 12.0
        if matchup == 0.0:
            score += 20.0

        if best is None or score > best[0]:
            best = (score, int(action), candidate, matchup, bulk_gain)

    # A lower threshold catches clear special-sponge upgrades such as
    # Celebi -> Blissey against a faster Starmie using only exact species
    # typing/base-stat evidence; hidden EVs and moves are still not invented.
    if best is None or best[0] < 14.0:
        return None, ""

    score, action, candidate, matchup, bulk_gain = best
    threat_kind = "special" if use_special else "physical"
    return action, (
        f"hidden-threat response: opponent has a faster high-{threat_kind}-offense profile "
        f"while the model selected {_key(selected)}; switch to {_key(candidate)} "
        f"(defensive matchup x{matchup:.1f}, bulk gain {bulk_gain:.0f}) before spending another passive turn"
    )
