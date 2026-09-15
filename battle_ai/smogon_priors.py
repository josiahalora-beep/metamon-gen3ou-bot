from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IV = 31
STAT_NAMES = ("hp", "atk", "def", "spa", "spd", "spe")
NATURE_MULTIPLIERS = {
    "adamant": {"atk": 1.1, "spa": 0.9}, "bold": {"def": 1.1, "atk": 0.9},
    "brave": {"atk": 1.1, "spe": 0.9}, "calm": {"spd": 1.1, "atk": 0.9},
    "careful": {"spd": 1.1, "spa": 0.9}, "gentle": {"spd": 1.1, "def": 0.9},
    "hasty": {"spe": 1.1, "def": 0.9}, "impish": {"def": 1.1, "spa": 0.9},
    "jolly": {"spe": 1.1, "spa": 0.9}, "lonely": {"atk": 1.1, "def": 0.9},
    "mild": {"spa": 1.1, "def": 0.9}, "modest": {"spa": 1.1, "atk": 0.9},
    "naive": {"spe": 1.1, "spd": 0.9}, "naughty": {"atk": 1.1, "spd": 0.9},
    "quiet": {"spa": 1.1, "spe": 0.9}, "rash": {"spa": 1.1, "spd": 0.9},
    "relaxed": {"def": 1.1, "spe": 0.9}, "sassy": {"spd": 1.1, "spe": 0.9},
    "timid": {"spe": 1.1, "atk": 0.9},
}


@dataclass(frozen=True)
class SetProfile:
    species: str
    name: str
    moves: tuple[str, ...]
    item: str
    nature: tuple[str, ...]
    evs: dict[str, int]
    weight: float
    evidence: tuple[str, ...]

    @property
    def confidence(self) -> float:
        return self.weight / max(1.0, self.weight + 1.0)


def _key(value: Any) -> str:
    return str(value).lower().replace(" ", "").replace("-", "").replace("_", "")


def _name(value: Any) -> str:
    return str(value).lower().replace(" ", "").replace("-", "").replace("_", "")


def _flatten_moves(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (_name(value),)
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(_flatten_moves(item))
        return tuple(dict.fromkeys(out))
    return ()


def _set_moves(raw_moves: Any) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_flatten_moves(raw_moves)))


def _set_natures(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        return (_key(raw),)
    if isinstance(raw, (list, tuple)):
        return tuple(_key(x) for x in raw if x)
    return ("neutral",)


def _load_raw(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}


def default_dataset_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "gen3ou.json"


def load_profiles(species: str, *, revealed_moves: tuple[str, ...] = (), path: Path | None = None) -> list[SetProfile]:
    data = _load_raw(path or default_dataset_path())
    target = _key(species)
    entry = next((value for key, value in data.items() if _key(key) == target), None)
    if not isinstance(entry, dict):
        return []

    revealed = {_name(x) for x in revealed_moves if x}
    ranked: list[SetProfile] = []
    for set_name, raw in entry.items():
        if not isinstance(raw, dict):
            continue
        raw_moves = raw.get("moves", [])
        all_moves = _set_moves(raw_moves)
        matches = sorted(move for move in revealed if move in all_moves)
        misses = sorted(move for move in revealed if move not in all_moves)

        # Preserve alternative hypotheses rather than deleting every set that
        # does not contain a revealed move. A revealed move strongly favors
        # compatible sets, while incompatible sets remain low-probability in
        # case the ladder opponent is running an uncommon/non-standard set.
        weight = 1.0 + 5.0 * len(matches)
        weight += 2.0 * len(matches) * len(matches)
        weight *= 0.12 ** len(misses)
        if len(matches) >= 2:
            weight += 3.0

        ranked.append(
            SetProfile(
                species=species,
                name=str(set_name),
                moves=all_moves,
                item=str(raw.get("item", "")),
                nature=_set_natures(raw.get("nature", "")),
                evs={str(k).lower(): int(v) for k, v in (raw.get("evs", {}) or {}).items()},
                weight=weight,
                evidence=tuple(matches),
            )
        )
    ranked.sort(key=lambda profile: (-profile.weight, profile.name))
    return ranked


def infer_profile(pokemon: Any, *, path: Path | None = None) -> SetProfile | None:
    revealed = tuple(
        _name(getattr(move, "id", getattr(move, "name", "")))
        for move in (getattr(pokemon, "moves", {}) or {}).values()
    )
    species = getattr(pokemon, "species", getattr(pokemon, "name", ""))
    profiles = load_profiles(str(species), revealed_moves=revealed, path=path)
    return profiles[0] if profiles else None


def profile_pool(pokemon: Any, *, path: Path | None = None, limit: int = 3) -> list[SetProfile]:
    revealed = tuple(
        _name(getattr(move, "id", getattr(move, "name", "")))
        for move in (getattr(pokemon, "moves", {}) or {}).values()
    )
    species = getattr(pokemon, "species", getattr(pokemon, "name", ""))
    return load_profiles(str(species), revealed_moves=revealed, path=path)[:limit]


def stat_from_profile(base_stat: int, stat: str, *, ev: int = 0, level: int = 100, iv: int = IV, nature: str = "neutral") -> int:
    stat = stat.lower()
    ev_term = max(0, min(255, int(ev))) // 4
    raw = ((2 * int(base_stat) + int(iv) + ev_term) * int(level)) // 100
    if stat == "hp":
        return raw + int(level) + 10
    value = raw + 5
    multiplier = NATURE_MULTIPLIERS.get(_key(nature), {}).get(stat, 1.0)
    return int(value * multiplier)


def expected_stats(pokemon: Any, profile: SetProfile | None) -> dict[str, int]:
    base = {str(k).lower(): int(v) for k, v in (getattr(pokemon, "base_stats", {}) or {}).items()}
    if not profile or not base:
        return {}
    level = int(getattr(pokemon, "level", 100) or 100)
    nature = profile.nature[0] if profile.nature else "neutral"
    return {
        stat: stat_from_profile(base.get(stat, 0), stat, ev=profile.evs.get(stat, 0), level=level, iv=IV, nature=nature)
        for stat in STAT_NAMES
    }


def likely_moves(pokemon: Any, *, path: Path | None = None, limit: int = 12) -> tuple[str, ...]:
    pool = profile_pool(pokemon, path=path, limit=8)
    weights: dict[str, float] = {}
    for profile in pool:
        for move in profile.moves:
            weights[move] = weights.get(move, 0.0) + profile.weight
    return tuple(move for move, _ in sorted(weights.items(), key=lambda item: (-item[1], item[0]))[:limit])
