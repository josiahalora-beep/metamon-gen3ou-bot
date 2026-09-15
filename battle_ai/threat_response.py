from __future__ import annotations

from typing import Any

from metamon.interface import consistent_pokemon_order
from .smogon_priors import expected_stats, infer_profile, likely_moves


_SUPER_EFFECTIVE = {
    "fire": {"grass", "ice", "bug", "steel"}, "water": {"fire", "ground", "rock"},
    "electric": {"water", "flying"}, "grass": {"water", "ground", "rock"},
    "ice": {"grass", "ground", "flying", "dragon"}, "fighting": {"normal", "ice", "rock", "dark", "steel"},
    "poison": {"grass"}, "ground": {"fire", "electric", "poison", "rock", "steel"},
    "flying": {"grass", "fighting", "bug"}, "psychic": {"fighting", "poison"},
    "bug": {"grass", "psychic", "dark"}, "rock": {"fire", "ice", "flying", "bug"},
    "ghost": {"ghost", "psychic"}, "dragon": {"dragon"}, "dark": {"psychic", "ghost"},
    "steel": {"ice", "rock"},
}
_RESISTED = {
    "fire": {"fire", "water", "rock", "dragon"}, "water": {"water", "grass", "dragon"},
    "electric": {"electric", "grass", "dragon"}, "grass": {"fire", "grass", "poison", "flying", "bug", "dragon", "steel"},
    "ice": {"fire", "water", "ice", "steel"}, "fighting": {"poison", "flying", "psychic", "bug"},
    "poison": {"poison", "ground", "rock", "ghost"}, "ground": {"grass", "bug"},
    "flying": {"electric", "rock", "steel"}, "psychic": {"steel", "psychic"},
    "bug": {"fire", "fighting", "poison", "flying", "ghost", "steel"}, "rock": {"fighting", "ground", "steel"},
    "ghost": {"dark"}, "dragon": {"steel"}, "dark": {"fighting", "dark", "steel"},
    "steel": {"fire", "water", "electric", "steel"},
}
_IMMUNE = {"normal": {"ghost"}, "fighting": {"ghost"}, "electric": {"ground"}, "poison": {"steel"},
           "ground": {"flying"}, "psychic": {"dark"}, "ghost": {"normal"}}


def _key(value: Any) -> str:
    return str(getattr(value, "name", getattr(value, "species", value))).lower().replace(" ", "").replace("-", "").replace("_", "")


def _types(pokemon: Any) -> tuple[str, ...]:
    return tuple(_key(t) for t in (getattr(pokemon, "types", ()) or ()) if t is not None)


def _fallback_stat(pokemon: Any, stat: str) -> float:
    stats = getattr(pokemon, "stats", {}) or {}
    try:
        value = stats.get(stat) if isinstance(stats, dict) else getattr(stats, stat, None)
        if value is not None:
            return float(value)
    except (TypeError, ValueError):
        pass
    base = getattr(pokemon, "base_stats", {}) or {}
    try:
        return float(base.get(stat, 0))
    except (AttributeError, TypeError, ValueError):
        return 0.0


def _profile_stat(pokemon: Any, stat: str) -> float:
    profile = infer_profile(pokemon)
    if profile is not None:
        values = expected_stats(pokemon, profile)
        if stat in values:
            return float(values[stat])
    return _fallback_stat(pokemon, stat)


def _boosted_stat(pokemon: Any, stat: str) -> float:
    value = _profile_stat(pokemon, stat)
    boosts = getattr(pokemon, "boosts", {}) or {}
    try:
        boost = int(boosts.get(stat, 0) or 0)
    except (TypeError, ValueError):
        boost = 0
    return value * ((2.0 + boost) / 2.0 if boost >= 0 else 2.0 / (2.0 - boost))


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
        return sorted(team, key=_key)


def _passive(move: Any) -> bool:
    return int(getattr(move, "base_power", 0) or 0) <= 0


def _revealed_moves(pokemon: Any) -> tuple[str, ...]:
    return tuple(_key(getattr(m, "id", getattr(m, "name", "")))
                   for m in (getattr(pokemon, "moves", {}) or {}).values())


def _revealed_move_types(pokemon: Any) -> tuple[str, ...]:
    out = []
    for move in (getattr(pokemon, "moves", {}) or {}).values():
        move_type = getattr(move, "type", None)
        if move_type is not None:
            out.append(_key(move_type))
    return tuple(dict.fromkeys(out))


def _threat_profile(pokemon: Any) -> tuple[bool, bool, str, tuple[str, ...]]:
    profile = infer_profile(pokemon)
    base = {str(k).lower(): int(v) for k, v in (getattr(pokemon, "base_stats", {}) or {}).items() if v is not None}
    base_spa = base.get("spa", 0)
    base_atk = base.get("atk", 0)
    fallback_special = base_spa >= max(90, base_atk + 10)
    fallback_physical = base_atk >= max(100, base_spa + 10)
    if profile is None:
        return fallback_special, fallback_physical, "species/base-stat prior", ()
    moves = set(profile.moves)
    special = {"surf", "hydropump", "thunderbolt", "thunder", "icebeam", "psychic", "fireblast", "flamethrower", "gigadrain", "hiddenpower"}
    physical = {"earthquake", "rockslide", "bodyslam", "return", "doubleedge", "focuspunch", "brickbreak", "explosion", "meteormash", "sludgebomb", "megahorn", "drillpeck"}
    spa_score = sum(m in special for m in moves) + (2 if profile.evs.get("spa", 0) >= 100 else 0)
    atk_score = sum(m in physical for m in moves) + (2 if profile.evs.get("atk", 0) >= 100 else 0)
    return spa_score >= atk_score or fallback_special, atk_score > spa_score or fallback_physical, profile.name, profile.evidence


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

    special_threat, physical_threat, profile_name, evidence = _threat_profile(opponent)
    active_speed = _boosted_stat(active, "spe")
    opponent_speed = _boosted_stat(opponent, "spe")
    opponent_base = {str(k).lower(): int(v) for k, v in (getattr(opponent, "base_stats", {}) or {}).items() if v is not None}
    active_base = {str(k).lower(): int(v) for k, v in (getattr(active, "base_stats", {}) or {}).items() if v is not None}
    faster = opponent_speed > active_speed * 1.03 or (
        opponent_base.get("spe", 0) > active_base.get("spe", 0)
        and opponent_base.get("spe", 0) >= active_base.get("spe", 0) + 10
    )
    if not faster or not (special_threat or physical_threat):
        return None, ""

    use_special = special_threat and not physical_threat
    bulk_stat = "spd" if use_special else "def"
    active_bulk = _profile_stat(active, "hp") + _boosted_stat(active, bulk_stat)
    revealed_types = _revealed_move_types(opponent)
    active_matchup = min((_type_multiplier(t, active) for t in revealed_types), default=1.0)

    slots = _switch_slots(battle)
    best = None
    for action in legal_actions:
        if action < 4:
            continue
        idx = int(action) - 4
        if not (0 <= idx < len(slots)):
            continue
        candidate = slots[idx]
        hp_fraction = float(getattr(candidate, "current_hp_fraction", 1.0) or 1.0)
        if getattr(candidate, "fainted", False) or hp_fraction < 0.35:
            continue
        candidate_bulk = _profile_stat(candidate, "hp") + _boosted_stat(candidate, bulk_stat)
        matchup = min((_type_multiplier(t, candidate) for t in revealed_types), default=1.0)
        bulk_gain = candidate_bulk - active_bulk
        matchup_gain = active_matchup - matchup
        candidate_base = {str(k).lower(): int(v) for k, v in (getattr(candidate, "base_stats", {}) or {}).items() if v is not None}

        score = max(0.0, bulk_gain) / 6.0 + matchup_gain * 55.0
        # Strong deterministic signal using exact Gen 3 base stats. This is
        # intentionally independent of guessed hidden moves.
        if use_special and candidate_base.get("spd", 0) >= opponent_base.get("spa", 0) and candidate_base.get("spd", 0) >= active_base.get("spd", 0) + 20:
            score += 30.0
        if use_special and candidate_base.get("spd", 0) >= 120:
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
    likely = ",".join(likely_moves(opponent, limit=6))
    return action, (
        f"hidden-threat response: {profile_name}; revealed={','.join(_revealed_moves(opponent)) or 'none'}; "
        f"likely={likely or 'none'}; 31-IV competitive-stat prior; faster {('special' if special_threat and not physical_threat else 'physical')} threat; "
        f"switch to {_key(candidate)} (matchup x{matchup:.1f}, bulk gain {bulk_gain:.0f})"
        + (f"; set evidence={','.join(evidence)}" if evidence else "")
    )
