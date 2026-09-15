from __future__ import annotations

from typing import Any


def _base_stats(obj: Any) -> dict[str, int]:
    raw = getattr(obj, "base_stats", {}) or {}
    if isinstance(raw, dict):
        return {str(k).lower(): int(v) for k, v in raw.items() if v is not None}
    return {}


def stat_range(obj: Any, stat: str, *, nature_min: float = 0.9, nature_max: float = 1.1) -> tuple[int, int] | None:
    """Conservative Gen 3 level-100 stat range when EVs/IVs are hidden."""
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
    """Estimate current HP when poke-env exposes the Gen 3 placeholder max_hp=100.

    The placeholder check is based on the observed HP representation rather
    than an arbitrary base-HP threshold, so 100-base-HP Pokémon such as Celebi
    and Swampert are handled correctly.
    """
    frac = getattr(obj, "current_hp_fraction", None)
    if frac is None:
        return None
    try:
        frac = float(frac)
    except (TypeError, ValueError):
        return None
    maximum = stat_range(obj, "hp")
    if maximum is None or frac <= 0:
        return maximum[0], maximum[0] if maximum is not None else None

    observed_max_hp = getattr(obj, "max_hp", None)
    try:
        observed_max_hp = int(observed_max_hp) if observed_max_hp is not None else None
    except (TypeError, ValueError):
        observed_max_hp = None

    # In the Gen 3 metamon/poke-env public battle path, unrevealed HP is
    # represented with max_hp=100. A genuine level-100 competitive OU HP pool
    # is above 100 for the Pokémon represented by this path, so use the species
    # HP envelope when the placeholder is present.
    placeholder = observed_max_hp == 100 and maximum[1] > 100
    if not placeholder:
        return None
    return max(1, int(maximum[0] * frac)), max(1, int(maximum[1] * frac))
