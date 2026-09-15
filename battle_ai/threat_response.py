from __future__ import annotations

from typing import Any

from metamon.interface import consistent_pokemon_order
from .smogon_priors import expected_stats, infer_profile, likely_moves


_SUPER_EFFECTIVE: dict[str, set[str]] = {
    "normal": set(), "fire": {"grass", "ice", "bug", "steel"}, "water": {"fire", "ground", "rock"},
    "electric": {"water", "flying"}, "grass": {"water", "ground", "rock"}, "ice": {"grass", "ground", "flying", "dragon"},
    "fighting": {"normal", "ice", "rock", "dark", "steel"}, "poison": {"grass"},
    "ground": {"fire", "electric", "poison", "rock", "steel"}, "flying": {"grass", "fighting", "bug"},
    "psychic": {"fighting", "poison"}, "bug": {"grass", "psychic", "dark"}, "rock": {"fire", "ice", "flying", "bug"},
    "ghost": {"ghost", "psychic"}, "dragon": {"dragon"}, "dark": {"psychic", "ghost"}, "steel": {"ice", "rock"},
}
_RESISTED: dict[str, set[str]] = {
    "normal": {"rock", "steel"}, "fire": {"fire", "water", "rock", "dragon"}, "water": {"water", "grass", "dragon"},
    "electric": {"electric", "grass", "dragon"}, "grass": {"fire", "grass", "poison", "flying", "bug", "dragon", "steel"},
    "ice": {"fire", "water", "ice", "steel"}, "fighting": {"poison", "flying", "psychic", "bug"},
    "poison": {"poison", "ground", "rock", "ghost"}, "ground": {"grass", "bug"}, "flying": {"electric", "rock", "steel"},
    "psychic": {"steel", "psychic"}, "bug": {"fire", "fighting", "poison", "flying", "ghost", "steel"},
    "rock": {"fighting", "ground", "steel"}, "ghost": {"dark"}, "dragon": {"steel"}, "dark": {"fighting", "dark", "steel"},
    "steel": {"fire", "water", "electric", "steel"},
}
_IMMUNE: dict[str, set[str]] = {
    "normal": {"ghost"}, "fighting": {"ghost"}, "electric": {"ground"}, "poison": {"steel"},
    "ground": {"flying"}, "psychic": {"dark"}, "ghost": {"normal"}, "dragon": set(),
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


def _fallback_stat(pokemon: Any, stat: str) -> float:
    stats = getattr(pokemon, "stats", {}) or {}
    try:
        value = stats.get(stat) if isinstance(stats, dict) else getattr(stats, stat, None)
        if value is not None:
            return float(value)
    except (TypeError, ValueError):
        pass
    return float(_base_stats(pokemon).get(stat, 0))


def _profile_stat(pokemon: Any, stat: str) -> float:
    profile = infer_profile(pokemon)
    if profile is not None:
        stats = expected_stats(pokemon, profile)
        if stat in stats:
            return float(stats[stat])
    return _fallback_stat(pokemon, stat)


def _boosted_stat(pokemon: Any, stat: str) -> float:
    value = _profile_stat(pokemon, stat)
    boosts = getattr(pokemon, "boosts", {}) or {}
    try:
        boost = int(boosts.get(stat, 0) or 0)
    except (TypeError, ValueError):
        boost = 0
    if boost >= 0:
        return value * ((2.0 + boost) / 2.0)
    return value * (2.0 / (2.0 - boost))


def _type_multiplier(attack_type: str, defender: Any) -> float:
    multiplier = 1.0
    for defense_type in _types(defender):
        if defense_type in _IMMUNE.get(attack_type, set()):
            return 0.0
        if defense_type in _SUPER_EFFECTIVE.get(attack_type, set()):
            multiplier *= 2.0
        elif defense_type in _RESISTED.get(attack_type, set()):
            multiplier *= 0.5
    return multiplier


def _switch_slots(battle: Any) -> list[Any]:
    team = [p for p in (getattr(battle, "team", {}) or {}).values()
            if not getattr(p, "fainted", False) and not getattr(p, "active", False)]
    try:
        return consistent_pokemon_order(team)
    except ValueError:
        return sorted(team, key=lambda p: _key(p))


def _passive(move: Any) -> bool:
    return int(getattr(move, "base_power", 0) or 0) <= 0


def _revealed_moves(pokemon: Any) -> tuple[str, ...]:
    return tuple(_key(getattr(move, "id", getattr(move, "name", "")))
                   for move in (getattr(pokemon, "moves", {}) or {}).values())


def _revealed_move_types(pokemon: Any) -> tuple[str, ...]:
    out: list[str] = []
    for move in (getattr(pokemon, "moves", {}) or {}).values():
        move_type = getattr(move, "type", None)
        if move_type is not None:
            out.append(_key(move_type))
    return tuple(dict.fromkeys(out))


def _profile_threat(pokemon: Any) -> tuple[bool, bool, str, str]:
    profile = infer_profile(pokemon)
    if profile is None:
        atk = _boosted_stat(pokemon, "atk")
        spa = _boosted_stat(pokemon, "spa")
        return spa >= atk and spa >= 90, atk > spa and atk >= 100, "species/base stats", ""

    moves = set(profile.moves)
    # These are broad Gen 3 offensive move families used only to classify the
    # inferred set. Exact damage still uses the dedicated Gen 3 calculator.
    special_names = {"surf", "hydropump", "thunderbolt", "thunder", "icebeam", "psychic", "fireblast", "flamethrower", "hiddenpower", "gigadrain"}
    physical_names = {"earthquake", "rockslide", "bodyslam", "return", "doubleedge", "focuspunch", "brickbreak", "explosion", "meteormash", "sludgebomb", "megahorn", "drillpeck", "extremespeed"}
    special_score = sum(1 for move in moves if move in special_names) + (2 if profile.evs.get("spa", 0) >= 100 else 0)
    physical_score = sum(1 for move in moves if move in physical_names) + (2 if profile.evs.get("atk", 0) >= 100 else 0)
    return special_score >= physical_score, physical_score > special_score, profile.name, ",".join(profile.evidence)


def hidden_threat_switch_override(battle: Any, legal_actions: list[int], model_action: int,
                                  move_map: dict[int, Any]) -> tuple[int | None, str]:
    if not (0 <= int(model_action) < 4):
        return None, ""
    selected = move_map.get(int(model_action))
    if selected is None or not _passive(selected):
        return None, ""

    active = getattr(battle, "active_pokemon", None)
    opponent = getattr(battle, "opponent_active_pokemon", None)
    if active is None or opponent is None or getattr(opponent, "fainted", False):
        return None, ""

    special_threat, physical_threat, profile_name, evidence = _profile_threat(opponent)
    active_speed = _boosted_stat(active, "spe")
    opponent_speed = _boosted_stat(opponent, "spe")
    if not (opponent_speed >= max(1.0, active_speed * 1.05)):
        return None, ""
    if not (special_threat or physical_threat):
        return None, ""

    use_special = special_threat and not physical_threat
    bulk_stat = "spd" if use_special else "def"
    active_bulk = _profile_stat(active, "hp") + _boosted_stat(active, bulk_stat)

    # Crucially, a Pokémon's own typing does not tell us what attacks it has.
    # Only revealed move types are used as hard coverage evidence. Hidden moves
    # remain uncertain and are represented by neutral pressure here.
    revealed_types = _revealed_move_types(opponent)
    if revealed_types:
        active_matchup = min(_type_multiplier(t, active) for t in revealed_types)
    else:
        active_matchup = 1.0

    slots = _switch_slots(battle)
    best: tuple[float, int, Any, float, float] | None = None
    for action in legal_actions:
        if action < 4:
            continue
        idx = int(action) - 4
        if not (0 <= idx < len(slots)):
            continue
        candidate = slots[idx]
        if float(getattr(candidate, "current_hp_fraction", 0.0) or 0.0) < 0.35:
            continue
        matchup = (min(_type_multiplier(t, candidate) for t in revealed_types)
                   if revealed_types else 1.0)
        bulk = _profile_stat(candidate, "hp") + _boosted_stat(candidate, bulk_stat)
        bulk_gain = bulk - active_bulk
        matchup_gain = active_matchup - matchup
        score = matchup_gain * 50.0 + max(0.0, bulk_gain) / 5.0
        if use_special and bulk_gain >= 40:
            score += 12.0
        if matchup <= 0.5:
            score += 18.0
        if matchup == 0.0:
            score += 30.0
        if best is None or score > best[0]:
            best = (score, int(action), candidate, matchup, bulk_gain)

    if best is None or best[0] < 10.0:
        return None, ""

    _, action, candidate, matchup, bulk_gain = best
    threat_kind = "special" if use_special else "physical"
    likely = ",".join(likely_moves(opponent, limit=6))
    return action, (
        f"hidden-threat response: {profile_name or 'species/base-stat prior'}; "
        f"revealed={','.join(_revealed_moves(opponent)) or 'none'}; "
        f"likely={likely or 'no cached set data'}; faster high-{threat_kind}-offense profile; "
        f"switch to {_key(candidate)} (matchup x{matchup:.1f}, bulk gain {bulk_gain:.0f}, "
        f"31-IV Smogon-stat prior)" + (f"; set evidence={evidence}" if evidence else "")
    )
