from __future__ import annotations

from typing import Any


def _base_stats(obj: Any) -> dict[str, int]:
    raw = getattr(obj, "base_stats", {}) or {}
    if isinstance(raw, dict):
        return {str(k).lower(): int(v) for k, v in raw.items() if v is not None}
    return {}


def stat_range(obj: Any, stat: str, *, nature_min: float = 0.9, nature_max: float = 1.1) -> tuple[int, int] | None:
    """Conservative Gen 3 level-100 stat range when EVs/IVs are hidden.

    The range intentionally spans legal 0..252 EV and 0..31 IV values and all
    possible nature multipliers. It is an uncertainty envelope, not a claim
    about the opponent's actual set.
    """
    stats = getattr(obj, "stats", {}) or {}
    if isinstance(stats, dict):
        known = stats.get(stat)
    else:
        known = getattr(stats, stat, None)
    if known is not None:
        value = int(known)
        return value, value

    base = _base_stats(obj).get(stat)
    if base is None:
        return None
    level = int(getattr(obj, "level", 100) or 100)
    level = max(1, level)
    if stat == "hp":
        low = ((2 * base + 0 + 0) * level // 100) + level + 10
        high = ((2 * base + 31 + 63) * level // 100) + level + 10
        return low, high
    low_raw = ((2 * base + 0 + 0) * level // 100) + 5
    high_raw = ((2 * base + 31 + 63) * level // 100) + 5
    return int(low_raw * nature_min), int(high_raw * nature_max)


def hp_range_from_observation(obj: Any) -> tuple[int, int] | None:
    """Estimate current HP range when poke-env exposes placeholder max_hp=100."""
    frac = getattr(obj, "current_hp_fraction", None)
    if frac is None:
        return None
    try:
        frac = float(frac)
    except (TypeError, ValueError):
        return None
    maximum = stat_range(obj, "hp")
    if maximum is None:
        return None
    if frac <= 0:
        return maximum[0], maximum[0]
    return max(1, int(maximum[0] * frac)), max(1, int(maximum[1] * frac))
