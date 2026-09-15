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


def _current_hp_range(obj: Any, hp_known: int) -> tuple[int, int] | None:
    """Return the defender's current HP range, not its maximum HP range."""
    current = getattr(obj, "current_hp", None)
    try:
        current = int(current) if current is not None else None
    except (TypeError, ValueError):
        current = None
    if current is not None and current >= 0:
        return current, current

    frac = getattr(obj, "current_hp_fraction", None)
    try:
        frac = float(frac) if frac is not None else None
    except (TypeError, ValueError):
        frac = None
    if frac is not None and 0.0 <= frac <= 1.0 and hp_known > 0:
        hp = max(0, int(round(hp_known * frac)))
        return hp, hp

    return (hp_known, hp_known) if hp_known > 0 else None


def _ivs(obj: Any) -> dict[str, int] | None:
    raw = getattr(obj, "ivs", None)
    if not isinstance(raw, dict):
        return None
    required = ("hp", "atk", "def", "spa", "spd", "spe")
    values: dict[str, int] = {}
    for key in required:
        try:
            value = raw.get(key)
            if value is None:
                return None
            values[key] = max(0, min(31, int(value)))
        except (TypeError, ValueError):
            return None
    return values


def _hidden_power_metadata(attacker: Any, move: Any) -> tuple[str | None, int | None, bool]:
    """Return the exact Gen 3 Hidden Power type/power when all IVs are known."""
    move_id = str(getattr(move, "id", getattr(move, "name", ""))).lower().replace(" ", "")
    if not move_id.startswith("hiddenpower"):
        return None, None, True
    ivs = _ivs(attacker)
    if ivs is None:
        return None, None, False

    # Gen 3 Hidden Power type and power use the low bits / full IV values.
    parity = (
        (ivs["hp"] & 1)
        + 2 * (ivs["atk"] & 1)
        + 4 * (ivs["def"] & 1)
        + 8 * (ivs["spe"] & 1)
        + 16 * (ivs["spa"] & 1)
        + 32 * (ivs["spd"] & 1)
    )
    type_index = math.floor(parity * 15 / 63)
    hp_types = ["fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel",
                "fire", "water", "grass", "electric", "psychic", "ice", "dragon", "dark"]
    iv_sum = (
        (ivs["hp"] % 4)
        + 4 * (ivs["atk"] % 4)
        + 16 * (ivs["def"] % 4)
        + 64 * (ivs["spe"] % 4)
        + 256 * (ivs["spa"] % 4)
        + 1024 * (ivs["spd"] % 4)
    )
    power = math.floor(iv_sum * 40 / 63) + 30
    return hp_types[type_index], power, True


def calculate_damage(attacker: Any, defender: Any, move: Any, *, weather: str = "",
                      critical: bool = False, reflect: bool = False,
                      light_screen: bool = False, random_rolls: int = 16) -> DamageRange:
    """Gen 3 damage with conservative hidden-stat estimation.

    Known stats use the exact cartridge-style calculation. If poke-env has not
    revealed a stat, the calculator uses a legal ADV stat envelope derived from
    species base stats, level, and observed HP fraction. It never substitutes
    fake stats such as 1, and it labels the result as estimated.

    Hidden Power is special in Gen 3: when all six IVs are available, its type
    and power are reconstructed from the IVs. For an opponent whose IVs are
    hidden, the move is deliberately treated as unreliable so tactical logic
    cannot turn an unknown Hidden Power into a false KO or safety fact.
    """
    power = int(getattr(move, "base_power", 0) or 0)
    move_type = str(getattr(getattr(move, "type", ""), "name", getattr(move, "type", ""))).lower()
    hp_type, hp_power, hp_known = _hidden_power_metadata(attacker, move)
    is_hidden_power = str(getattr(move, "id", getattr(move, "name", ""))).lower().replace(" ", "").startswith("hiddenpower")
    if is_hidden_power and hp_known:
        if hp_power is None or hp_type is None:
            return DamageRange(0, 0, 0.0, 0.0, 0.0, reliable=False, reason="missing Hidden Power IV metadata")
        move_type = hp_type
        power = hp_power
    elif is_hidden_power and not hp_known:
        return DamageRange(0, 0, 0.0, 0.0, 0.0, reliable=False, reason="Hidden Power IVs are hidden")

    if power <= 0:
        return DamageRange(0, 0, 0.0, 0.0, 0.0)
    physical_types = {"normal", "fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel"}
    physical = move_type in physical_types
    atk_key, def_key = ("atk", "def") if physical else ("spa", "spd")

    atk_range = _known_or_estimated(attacker, atk_key)
    def_range = _known_or_estimated(defender, def_key)
    if atk_range is None or def_range is None:
        return DamageRange(0, 0, 0.0, 0.0, 0.0, reliable=False, reason="missing battle stats and species base stats")

    hp_known_value = getattr(defender, "max_hp", None)
    try:
        hp_known_value = int(hp_known_value) if hp_known_value is not None else 0
    except (TypeError, ValueError):
        hp_known_value = 0
    hp_estimate = hp_range_from_observation(defender)
    placeholder_hp = hp_known_value == 100 and hp_estimate is not None
    if placeholder_hp:
        hp_range = hp_estimate
    else:
        hp_range = _current_hp_range(defender, hp_known_value)
        if hp_range is None:
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

    values = [one(atk, defense, roll) for atk in (atk_lo, atk_hi)
              for defense in (def_lo, def_hi)
              for roll in range(217, 256)]
    min_damage = min(values)
    max_damage = max(values)
    hp_lo, hp_hi = hp_range
    estimated = atk_range[0] != atk_range[1] or def_range[0] != def_range[1] or hp_lo != hp_hi
    if estimated:
        ko_probability = 1.0 if min_damage >= hp_hi else 0.0
    else:
        ko_probability = sum(v >= hp_lo for v in values) / len(values)
    pct_min = 100.0 * min_damage / max(1, hp_known_value or hp_hi)
    pct_max = 100.0 * max_damage / max(1, hp_known_value or hp_lo)
    reason = "estimated from species/base stats and observed HP" if estimated else ""
    return DamageRange(min_damage, max_damage, pct_min, pct_max, ko_probability, reliable=True, reason=reason)
