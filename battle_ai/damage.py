from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

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


def _stat(obj: Any, key: str, default: int = 1) -> int:
    if isinstance(obj, dict):
        return int(obj.get(key, default) or default)
    return int(getattr(obj, key, default) or default)


def _boost(value: int) -> float:
    return (2 + value) / 2 if value >= 0 else 2 / (2 - value)


def type_multiplier(move_type: str, defender_types: list[str] | tuple[str, ...]) -> float:
    mult = 1.0
    for defender_type in defender_types:
        entry = _TYPES.get(str(defender_type).lower(), {}).get("damageTaken", {})
        code = entry.get(str(move_type).capitalize(), entry.get(str(move_type).lower(), 0))
        mult *= {0: 1.0, 1: 2.0, 2: 0.5, 3: 0.0}.get(code, 1.0)
    return mult


def calculate_damage(attacker: Any, defender: Any, move: Any, *, weather: str = "",
                      critical: bool = False, reflect: bool = False,
                      light_screen: bool = False, random_rolls: int = 16) -> DamageRange:
    """Gen 3 damage range using the cartridge-style 217..255 random factor."""
    power = int(getattr(move, "base_power", 0) or 0)
    if power <= 0:
        return DamageRange(0, 0, 0.0, 0.0, 0.0)
    move_type = str(getattr(getattr(move, "type", ""), "name", getattr(move, "type", ""))).lower()
    physical_types = {"normal", "fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel"}
    physical = move_type in physical_types
    atk_key, def_key = ("atk", "def") if physical else ("spa", "spd")
    raw_atk = (getattr(attacker, "stats", attacker) or {}).get(atk_key) if isinstance(getattr(attacker, "stats", attacker), dict) else getattr(getattr(attacker, "stats", attacker), atk_key, None)
    raw_defense = (getattr(defender, "stats", defender) or {}).get(def_key) if isinstance(getattr(defender, "stats", defender), dict) else getattr(getattr(defender, "stats", defender), def_key, None)
    raw_hp = getattr(defender, "max_hp", None)
    if raw_atk is None or raw_defense is None or raw_hp is None or int(raw_hp or 0) <= 0:
        return DamageRange(0, 0, 0.0, 0.0, 0.0, reliable=False, reason="missing battle stats or HP")
    atk = int(raw_atk)
    defense = int(raw_defense)
    boosts_a = getattr(attacker, "boosts", {}) or {}
    boosts_d = getattr(defender, "boosts", {}) or {}
    atk_stage = int(boosts_a.get(atk_key, 0))
    def_stage = int(boosts_d.get(def_key, 0))
    # Gen 3 critical hits ignore negative offensive and positive defensive
    # stages, but retain the opposite stages.
    if critical:
        atk_stage = max(0, atk_stage)
        def_stage = min(0, def_stage)
    atk = max(1, int(atk * _boost(atk_stage)))
    defense = max(1, int(defense * _boost(def_stage)))
    if str(getattr(move, "id", "")).lower() in {"explosion", "selfdestruct"} and physical:
        defense = max(1, math.floor(defense / 2))
    level = int(getattr(attacker, "level", 100) or 100)
    base = math.floor(math.floor(math.floor(2 * level / 5 + 2) * power * atk / defense) / 50) + 2
    status = getattr(attacker, "status", "")
    status = str(getattr(status, "name", status)).lower()
    if weather.lower() in {"raindance", "rain"}:
        base = math.floor(base * (1.5 if move_type == "water" else 0.5 if move_type == "fire" else 1.0))
    elif weather.lower() in {"sunnyday", "sun"}:
        base = math.floor(base * (1.5 if move_type == "fire" else 0.5 if move_type == "water" else 1.0))
    if critical:
        base = math.floor(base * 2)
    stab = 1.5 if move_type in {str(getattr(x, "name", x)).lower() for x in getattr(attacker, "types", ())} else 1.0
    defender_types = list(getattr(defender, "types", ()))
    values = []
    for roll in range(217, 256):
        damage = math.floor(base * roll / 255)
        damage = math.floor(damage * stab)
        immune = False
        for defender_type in defender_types:
            entry = _TYPES.get(str(getattr(defender_type, "name", defender_type)).lower(), {}).get("damageTaken", {})
            code = entry.get(str(move_type).capitalize(), entry.get(str(move_type).lower(), 0))
            if code == 3:
                immune = True
                damage = 0
                break
            if code == 1:
                damage *= 2
            elif code == 2:
                damage = math.floor(damage / 2)
        if status in {"brn", "burn"} and physical:
            damage = math.floor(damage / 2)
        if reflect and physical and not critical:
            damage = math.floor(damage / 2)
        if light_screen and not physical and not critical:
            damage = math.floor(damage / 2)
        values.append(0 if immune else max(1, damage))
    hp = int(raw_hp)
    ko = sum(v >= hp for v in values) / max(1, len(values))
    return DamageRange(min(values), max(values), 100 * min(values) / hp, 100 * max(values) / hp, ko)
