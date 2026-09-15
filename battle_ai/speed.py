from __future__ import annotations

import math
from typing import Any


def _stat(p: Any) -> int:
    stats = getattr(p, "stats", p) or {}
    return int(stats.get("spe", 1) if isinstance(stats, dict) else getattr(stats, "spe", 1))


def speed_range(pokemon: Any, *, scarf: bool | None = None) -> tuple[int, int]:
    """Return an uncertainty interval; known poke-env stats produce a point range."""
    value = _stat(pokemon)
    item = str(getattr(pokemon, "item", "")).lower()
    if scarf is True or (scarf is None and "choicescarf" in item):
        value = math.floor(value * 1.5)
    status = str(getattr(pokemon, "status", "")).lower()
    if status in {"par", "paralysis"}:
        value = math.floor(value / 4)
    return value, value


def can_outspeed(attacker: Any, defender: Any) -> bool:
    return speed_range(attacker)[0] > speed_range(defender)[1]


def speed_tie_probability(a: Any, b: Any) -> float:
    ar = speed_range(a); br = speed_range(b)
    if ar[0] > br[1]: return 1.0
    if br[0] > ar[1]: return 0.0
    return 0.5
