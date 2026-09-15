from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from .estimation import stat_range, hp_range_from_observation

_ROOT = Path(__file__).resolve().parents[1] / "metamon" / "backend" / "showdown_dex" / "static"
_TYPES = json.loads((_ROOT / "typechart" / "gen3typechart.json").read_text())


@dataclass(frozen=True)
class DamageRange:
    min_damage: int
    max_damage: int
    percentage_min: float
    percentage_max: float
    ko_probability: float
    reliable: bool = True
    reason: str = ""


def _boost(value: int) -> float:
    return (2 + value) / 2 if value >= 0 else 2 / (2 - value)


def type_multiplier(move_type: str, defender_types: list[str] | tuple[str, ...]) -> float:
    mult = 1.0
    for defender_type in defender_types:
        entry = _TYPES.get(str(getattr(defender_type, "name", defender_type)).lower(), {}).get("damageTaken", {})
        code = entry.get(str(move_type).capitalize(), entry.get(str(move_type).lower(), 0))
        mult *= {0: 1.0, 1: 2.0, 2: 0.5, 3: 0.0}.get(code, 1.0)
    return mult


def _known_or_estimated(obj: Any, stat: str) -> tuple[int, int] | None:
    return stat_range(obj, stat)


def calculate_damage(attacker: Any, defender: Any, move: Any, *, weather: str = "",
                      critical: bool = False, reflect: bool = False,
                      light_screen: bool = False, random_rolls: int = 16) -> DamageRange:
    """Gen 3 damage with conservative hidden-stat estimation.

    Known stats use the exact cartridge-style calculation. If poke-env has not
    revealed a stat, the calculator uses a legal ADV stat envelope derived from
    species base stats, level, and observed HP fraction. It never substitutes
    fake stats such as 1, and it labels the result as estimated.
    """
    power = int(getattr(move, "base_power", 0) or 0)
    if power <= 0:
        return DamageRange(0, 0, 0.0, 0.0, 0.0)
    move_type = str(getattr(getattr(move, "type", ""), "name", getattr(move, "type", ""))).lower()
    physical_types = {"normal", "fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel"}
    physical = move_type in physical_types
    atk_key, def_key = ("atk", "def") if physical else ("spa", "spd")

    atk_range = _known_or_estimated(attacker, atk_key)
    def_range = _known_or_estimated(defender, def_key)
    if atk_range is None or def_range is None:
        return DamageRange(0, 0, 0.0, 0.0, 0.0, reliable=False, reason="missing battle stats and species base stats")

    hp_known = getattr(defender, "max_hp", None)
    try:
        hp_known = int(hp_known) if hp_known is not None else 0
    except (TypeError, ValueError):
        hp_known = 0
    hp_fraction = float(getattr(defender, "current_hp_fraction", 0.0) or 0.0)
    hp_estimate = hp_range_from_observation(defender)
    placeholder_hp = hp_known <= 100 and hp_fraction > 0 and int(getattr(defender, "base_stats", {}).get("hp", 0) or 0) >= 150
    if hp_known > 0 and not placeholder_hp:
        hp_range = (hp_known, hp_known)
    elif hp_estimate is not None:
        hp_range = hp_estimate
    else:
        return DamageRange(0, 0, 0.0, 0.0, 0.0, reliable=False, reason="missing battle HP")

    boosts_a = getattr(attacker, "boosts", {}) or {}
    boosts_d = getattr(defender, "boosts", {}) or {}
    atk_stage = int(boosts_a.get(atk_key, 0) or 0)
    def_stage = int(boosts_d.get(def_key, 0) or 0)
    if critical:
        atk_stage = max(0, atk_stage)
        def_stage = min(0, def_stage)
    atk_lo = max(1, int(atk_range[0] * _boost(atk_stage)))
    atk_hi = max(1, int(atk_range[1] * _boost(atk_stage)))
    def_lo = max(1, int(def_range[0] * _boost(def_stage)))
    def_hi = max(1, int(def_range[1] * _boost(def_stage)))
    if str(getattr(move, "id", "")).lower() in {"explosion", "selfdestruct"} and physical:
        def_lo = max(1, math.floor(def_lo / 2))
        def_hi = max(1, math.floor(def_hi / 2))

    level = int(getattr(attacker, "level", 100) or 100)
    level = max(1, level)
    stab = 1.5 if move_type in {str(getattr(x, "name", x)).lower() for x in getattr(attacker, "types", ())} else 1.0
    defender_types = list(getattr(defender, "types", ()))
    mult = type_multiplier(move_type, defender_types)
    if mult == 0:
        return DamageRange(0, 0, 0.0, 0.0, 0.0, reliable=True, reason="type immunity")

    def one(atk: int, defense: int, roll: int) -> int:
        base = math.floor(math.floor(math.floor(2 * level / 5 + 2) * power * atk / defense) / 50) + 2
        weather_name = weather.lower()
        if weather_name in {"raindance", "rain"}:
            base = math.floor(base * (1.5 if move_type == "water" else 0.5 if move_type == "fire" else 1.0))
        elif weather_name in {"sunnyday", "sun"}:
            base = math.floor(base * (1.5 if move_type == "fire" else 0.5 if move_type == "water" else 1.0))
        if critical:
            base = math.floor(base * 2)
        damage = math.floor(base * roll / 255)
        damage = math.floor(damage * stab * mult)
        status = str(getattr(getattr(attacker, "status", ""), "name", getattr(attacker, "status", ""))).lower()
        if status in {"brn", "burn"} and physical:
            damage = math.floor(damage / 2)
        if reflect and physical and not critical:
            damage = math.floor(damage / 2)
        if light_screen and not physical and not critical:
            damage = math.floor(damage / 2)
        return max(1, damage)

    # Worst/best legal envelopes. For an unknown opponent, the range is meant
    # to answer tactical questions without pretending we know its EV spread.
    values = [one(atk, defense, roll) for atk in (atk_range[0], atk_range[1])
              for defense in (def_range[0], def_range[1])
              for roll in range(217, 256)]
    min_damage = min(values)
    max_damage = max(values)
    hp_lo, hp_hi = hp_range
    # KO probability is deliberately conservative for estimated HP/stat data:
    # 1 only when every legal envelope member KOs, otherwise 0. Known data keeps
    # the exact random-roll probability.
    estimated = atk_range[0] != atk_range[1] or def_range[0] != def_range[1] or hp_lo != hp_hi
    if estimated:
        ko_probability = 1.0 if min_damage >= hp_hi else 0.0
    else:
        ko_probability = sum(v >= hp_lo for v in values) / len(values)
    pct_min = 100.0 * min_damage / hp_hi
    pct_max = 100.0 * max_damage / hp_lo
    reason = "estimated from species/base stats and observed HP" if estimated else ""
    return DamageRange(min_damage, max_damage, pct_min, pct_max, ko_probability, reliable=True, reason=reason)
