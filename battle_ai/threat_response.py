from __future__ import annotations

from typing import Any

from metamon.interface import consistent_pokemon_order

from .smogon_priors import expected_stats, infer_profile, likely_moves


# Gen 3 type chart compressed for defensive-switch reasoning.
_SUPER_EFFECTIVE: dict[str, set[str]] = {
    "normal": set(), "fire": {"grass", "ice", "bug", "steel"},
    "water": {"fire", "ground", "rock"}, "electric": {"water", "flying"},
    "grass": {"water", "ground", "rock"}, "ice": {"grass", "ground", "flying", "dragon"},
    "fighting": {"normal", "ice", "rock", "dark", "steel"}, "poison": {"grass"},
    "ground": {"fire", "electric", "poison", "rock", "steel"},
    "flying": {"grass", "fighting", "bug"}, "psychic": {"fighting", "poison"},
    "bug": {"grass", "psychic", "dark"}, "rock": {"fire", "ice", "flying", "bug"},
    "ghost": {"ghost", "psychic"}, "dragon": {"dragon"}, "dark": {"psychic", "ghost"},
    "steel": {"ice", "rock"},
}
_RESISTED: dict[str, set[str]] = {
    "normal": {"rock", "steel"}, "fire": {"fire", "water", "rock", "dragon"},
    "water": {"water", "grass", "dragon"}, "electric": {"electric", "grass", "dragon"},
    "grass": {"fire", "grass", "poison", "flying", "bug", "dragon", "steel"},
    "ice": {"fire", "water", "ice", "steel"}, "fighting": {"poison", "flying", "psychic", "bug"},
    "poison": {"poison", "ground", "rock", "ghost"}, "ground": {"grass", "bug"},
    "flying": {"electric", "rock", "steel"}, "psychic": {"steel", "psychic"},
    "bug": {"fire", "fighting", "poison", "flying", "ghost", "steel"},
    "rock": {"fighting", "ground", "steel"}, "ghost": {"dark"}, "dragon": {"steel"},
    "dark": {"fighting", "dark", "steel"}, "steel": {"fire", "water", "electric", "steel"},
}
_IMMUNE: dict[str, set[str]] = {
    "normal": {"ghost"}, "fighting": {"ghost"}, "electric": {"ground"},
    "poison": {"steel"}, "ground": {"flying"}, "psychic": {"dark"},
    "ghost": {"normal"}, "dragon": set(),
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
    return _stat(pokemon, stat)


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


def _revealed_moves(pokemon: Any) -> tuple[str, ...]:
    return tuple(
        _key(getattr(move, "id", getattr(move, "name", "")))
        for move in (getattr(pokemon, "moves", {}) or {}).values()
    )


def _profile_threat(pokemon: Any) -> tuple[bool, bool, str, str]:
    """Return special/physical threat flags using the inferred competitive set prior."""
    profile = infer_profile(pokemon)
    if profile is None:
        atk = _boosted_stat(pokemon, "atk")
        spa = _boosted_stat(pokemon, "spa")
        special = spa >= atk and spa >= 90
        physical = atk > spa and atk >= 100
        return special, physical, "species/base stats", ""

    moves = set(profile.moves)
    special_names = {"surf", "hydropump", "thunderbolt", "thunder", "icebeam", "psychic", "fireblast", "flamethrower", "hiddenpower", "gigadrain"}
    physical_names = {"earthquake", "rockslide", "rockslide", "bodyslam", "return", "doubleedge", "focuspunch", "brickbreak", "hiddenpowerghost", "explosion", "meteormash", "sludgebomb", "megahorn", "drillpeck", "extremespeed"}
    special_score = sum(1 for move in moves if move in special_names) + (2 if "spa" in profile.evs else 0)
    physical_score = sum(1 for move in moves if move in physical_names) + (2 if "atk" in profile.evs else 0)
    spa = _boosted_stat(pokemon, "spa")
    atk = _boosted_stat(pokemon, "atk")
    return (
        special_score >= physical_score,
        physical_score > special_score,
        profile.name,
        ",".join(profile.evidence),
    )


def hidden_threat_switch_override(
    battle: Any,
    legal_actions: list[int],
    model_action: int,
    move_map: dict[int, Any],
) -> tuple[int | None, str]:
    """Use exact Gen 3 species data plus a Smogon ADV OU set posterior.

    Unrevealed moves are probabilities, not facts. A revealed move sharply raises
    compatible Smogon sets, while 31-IV competitive stat reconstruction supplies
    a concrete expected speed/offense profile for the candidate set.
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

    special_threat, physical_threat, profile_name, evidence = _profile_threat(opponent)
    active_speed = _boosted_stat(active, "spe")
    opponent_speed = _boosted_stat(opponent, "spe")
    speed_threat = opponent_speed >= active_speed * 1.05
    if not (special_threat or physical_threat) or not speed_threat:
        return None, ""

    use_special = special_threat and not physical_threat
    bulk_stat = "spd" if use_special else "def"
    active_bulk = _profile_stat(active, "hp") + _boosted_stat(active, bulk_stat)

    opponent_types = _types(opponent)
    active_matchup = 1.0
    for attack_type in opponent_types:
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
        if _profile_stat(candidate, "hp") <= 0 or float(getattr(candidate, "current_hp_fraction", 0.0) or 0.0) < 0.50:
            continue

        matchup = 1.0
        for attack_type in opponent_types:
            matchup = min(matchup, _type_multiplier(attack_type, candidate))
        bulk = _profile_stat(candidate, "hp") + _boosted_stat(candidate, bulk_stat)
        bulk_gain = bulk - active_bulk
        matchup_gain = active_matchup - matchup
        score = matchup_gain * 45.0 + max(0.0, bulk_gain) / 5.0
        if use_special and bulk_gain > 25:
            score += 10.0
        if matchup <= 0.5:
            score += 14.0
        if matchup == 0.0:
            score += 25.0
        if best is None or score > best[0]:
            best = (score, int(action), candidate, matchup, bulk_gain)

    if best is None or best[0] < 12.0:
        return None, ""

    _, action, candidate, matchup, bulk_gain = best
    likely = ",".join(likely_moves(opponent, limit=6))
    threat_kind = "special" if use_special else "physical"
    evidence_text = f"; set evidence={evidence}" if evidence else ""
    return action, (
        f"hidden-threat response: {profile_name or 'species/base-stat prior'}; "
        f"revealed={','.join(_revealed_moves(opponent)) or 'none'}; "
        f"likely={likely or 'no cached set data'}; "
        f"faster high-{threat_kind}-offense profile; switch to {_key(candidate)} "
        f"(matchup x{matchup:.1f}, bulk gain {bulk_gain:.0f}, 31-IV Smogon-stat prior)"
        f"{evidence_text}"
    )
