from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


def _name(value: Any) -> str:
    if value is None:
        return ""
    return str(getattr(value, "name", value)).lower().replace(" ", "")


def _pokemon(p: Any) -> dict:
    if p is None:
        return {}
    moves = []
    for move in getattr(p, "moves", {}).values():
        moves.append({"name": _name(move), "id": _name(getattr(move, "id", "")),
                      "type": _name(getattr(move, "type", "")),
                      "category": _name(getattr(move, "category", "")),
                      "base_power": int(getattr(move, "base_power", 0) or 0),
                      "priority": int(getattr(move, "priority", 0) or 0)})
    return {
        "name": _name(getattr(p, "species", getattr(p, "name", ""))),
        "base_species": _name(getattr(p, "base_species", "")),
        "types": [_name(t) for t in getattr(p, "types", ())],
        "hp": int(getattr(p, "current_hp", 0) or 0),
        "max_hp": int(getattr(p, "max_hp", 0) or 0),
        "hp_fraction": float(getattr(p, "current_hp_fraction", 0.0) or 0.0),
        "status": _name(getattr(p, "status", "")),
        "item": _name(getattr(p, "item", "")),
        "ability": _name(getattr(p, "ability", "")),
        "level": int(getattr(p, "level", 100) or 100),
        "base_stats": dict(getattr(p, "base_stats", {}) or {}),
        "stats": dict(getattr(p, "stats", {}) or {}),
        "boosts": dict(getattr(p, "boosts", {}) or {}),
        "moves": moves,
        "fainted": bool(getattr(p, "fainted", False)),
    }


@dataclass(frozen=True)
class BattleState:
    battle_id: str
    turn: int
    format: str
    player: str
    opponent: str
    our_active: dict
    opponent_active: dict
    our_team: tuple[dict, ...]
    opponent_team: tuple[dict, ...]
    available_actions: tuple[int, ...]
    weather: tuple[str, ...]
    our_hazards: dict
    opponent_hazards: dict
    field: tuple[str, ...]
    forced_switch: bool

    def to_dict(self) -> dict:
        return asdict(self)


def snapshot_battle(battle: Any, available_actions=()) -> BattleState:
    team = tuple(_pokemon(p) for p in getattr(battle, "team", {}).values())
    opp = tuple(_pokemon(p) for p in getattr(battle, "opponent_team", {}).values())
    return BattleState(
        battle_id=str(getattr(battle, "battle_tag", "unknown")),
        turn=int(getattr(battle, "turn", 0) or 0),
        format=str(getattr(battle, "format", "gen3ou")),
        player=str(getattr(battle, "player_username", "")),
        opponent=str(getattr(battle, "opponent_username", "")),
        our_active=_pokemon(getattr(battle, "active_pokemon", None)),
        opponent_active=_pokemon(getattr(battle, "opponent_active_pokemon", None)),
        our_team=team,
        opponent_team=opp,
        available_actions=tuple(int(x) for x in available_actions),
        weather=tuple(_name(x) for x in getattr(battle, "weather", {}).keys()),
        our_hazards={_name(k): int(v) for k, v in (getattr(battle, "side_conditions", {}) or {}).items()},
        opponent_hazards={_name(k): int(v) for k, v in (getattr(battle, "opponent_side_conditions", {}) or {}).items()},
        field=tuple(_name(x) for x in (getattr(battle, "fields", {}) or {}).keys()),
        forced_switch=bool(getattr(battle, "force_switch", False)),
    )
