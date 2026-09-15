from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from .damage import calculate_damage, type_multiplier
from metamon.interface import consistent_move_order, consistent_pokemon_order


_SETUP_BOOSTS: dict[str, dict[str, int]] = {
    "dragondance": {"atk": 1, "spe": 1},
    "calmmind": {"spa": 1, "spd": 1},
    "swordsdance": {"atk": 2},
    "agility": {"spe": 2},
    "rockpolish": {"spe": 2},
    "curse": {"atk": 1, "def": 1, "spe": -1},
    "bellydrum": {"atk": 6},
}
_SETUP_MOVES = set(_SETUP_BOOSTS)
_PHAZING = {"roar", "whirlwind", "haze"}
_GROUNDED_IMMUNE_TYPES = {"flying"}


@dataclass(frozen=True)
class StrategicPlan:
    primary_win_condition: str | None
    setup_move: str | None
    sweep_ready: bool
    sweep_ko_count: int
    hazard_goal: int
    hazard_layer: int
    cleanup_mode: bool
    required_removals: tuple[str, ...]
    reason: str


def _move_id(move: Any) -> str:
    return str(getattr(move, "id", getattr(move, "name", ""))).lower().replace(" ", "")


def _base_power(move: Any) -> int:
    try:
        return int(getattr(move, "base_power", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _species_key(pokemon: Any) -> str:
    value = getattr(pokemon, "species", getattr(pokemon, "name", ""))
    return str(value).lower().replace(" ", "").replace("-", "")


def _hp_fraction(pokemon: Any) -> float:
    return float(getattr(pokemon, "current_hp_fraction", 0.0) or 0.0)


def _weather(battle: Any) -> str:
    return ",".join(
        str(getattr(x, "name", x)).lower()
        for x in (getattr(battle, "weather", {}) or {}).keys()
    )


def _ordered_moves(pokemon: Any) -> list[Any]:
    moves = list((getattr(pokemon, "moves", {}) or {}).values())
    try:
        return consistent_move_order(moves)
    except ValueError:
        return sorted(moves, key=_move_id)


def _switch_slots(battle: Any) -> list[Any]:
    team = [
        p for p in (getattr(battle, "team", {}) or {}).values()
        if not getattr(p, "fainted", False) and not getattr(p, "active", False)
    ]
    try:
        return consistent_pokemon_order(team)
    except ValueError:
        return sorted(team, key=_species_key)


def _team_alive(battle: Any) -> list[Any]:
    return [p for p in (getattr(battle, "team", {}) or {}).values() if not getattr(p, "fainted", False)]


def _revealed_opponents(battle: Any) -> list[Any]:
    return [p for p in (getattr(battle, "opponent_team", {}) or {}).values() if not getattr(p, "fainted", False)]


def _is_grounded(pokemon: Any) -> bool:
    types = {str(getattr(t, "name", t)).lower() for t in (getattr(pokemon, "types", ()) or ())}
    if types & _GROUNDED_IMMUNE_TYPES:
        return False
    ability = str(getattr(pokemon, "ability", "") or "").lower().replace(" ", "")
    return ability != "levitate"


def _has_revealed_move(pokemon: Any, ids: set[str]) -> bool:
    return any(_move_id(m) in ids for m in (getattr(pokemon, "moves", {}) or {}).values())


def _incoming_safe(battle: Any, defender: Any, opponent: Any) -> tuple[bool, float | None, str]:
    if defender is None or opponent is None:
        return False, None, "missing combatants"
    moves = [m for m in (getattr(opponent, "moves", {}) or {}).values() if _base_power(m) > 0]
    if not moves:
        return True, None, "no revealed damaging move"
    reliable = []
    for move in moves:
        result = calculate_damage(opponent, defender, move, weather=_weather(battle))
        if result.reliable:
            reliable.append((result.percentage_max, _move_id(move)))
    if not reliable:
        return False, None, "revealed incoming damage is not reliable enough"
    worst_pct, worst_move = max(reliable, key=lambda item: item[0])
    current_hp_pct = _hp_fraction(defender) * 100.0
    # A setup or hazard turn needs a margin for rolls/status and the following
    # turn. The threshold is intentionally stricter than merely surviving.
    safe = worst_pct < max(18.0, current_hp_pct * 0.55)
    return safe, worst_pct, worst_move


def _boosted_clone(active: Any, boosts: dict[str, int]) -> Any | None:
    try:
        clone = copy.copy(active)
        current = dict(getattr(active, "boosts", {}) or {})
        for stat, delta in boosts.items():
            current[stat] = max(-6, min(6, int(current.get(stat, 0) or 0) + delta))
        clone.boosts = current
        return clone
    except Exception:
        return None


def _best_damage_ko(attacker: Any, defender: Any, *, weather: str) -> tuple[float, int | None, Any | None]:
    best_ko = 0.0
    best_idx = None
    best_move = None
    for idx, move in enumerate(_ordered_moves(attacker)[:4]):
        if _base_power(move) <= 0:
            continue
        result = calculate_damage(attacker, defender, move, weather=weather)
        if result.reliable and result.ko_probability > best_ko:
            best_ko = result.ko_probability
            best_idx = idx
            best_move = move
    return best_ko, best_idx, best_move


def _candidate_sweep_value(battle: Any, candidate: Any, setup_move: Any) -> tuple[float, int, str, tuple[str, ...]]:
    if candidate is None or _hp_fraction(candidate) < 0.60:
        return 0.0, 0, "candidate too compromised", ()

    revealed = _revealed_opponents(battle)
    if not revealed:
        return 0.0, 0, "no revealed targets", ()

    opponent_ids = {
        _move_id(m)
        for foe in revealed
        for m in (getattr(foe, "moves", {}) or {}).values()
    }
    if opponent_ids & _PHAZING:
        return 0.0, 0, "revealed phazing prevents a reliable setup plan", ()

    clone = _boosted_clone(candidate, _SETUP_BOOSTS[_move_id(setup_move)])
    if clone is None:
        return 0.0, 0, "could not simulate boosted candidate", ()

    weather = _weather(battle)
    boosted_kos = 0
    meaningful_gain = 0.0
    required_removals: list[str] = []

    for foe in revealed:
        before_ko, _, _ = _best_damage_ko(candidate, foe, weather=weather)
        after_ko, _, after_move = _best_damage_ko(clone, foe, weather=weather)
        if after_ko >= 1.0:
            boosted_kos += 1
        meaningful_gain += max(0.0, after_ko - before_ko)
        if after_ko < 1.0:
            required_removals.append(_species_key(foe))

    # A genuine sweep window means the boosted candidate can immediately KO
    # multiple visible targets. One-target setup is accepted only when it
    # creates the current guaranteed KO and the candidate is otherwise safe.
    current_ko, _, _ = _best_damage_ko(candidate, getattr(battle, "opponent_active_pokemon", None), weather=weather)
    boosted_current_ko, _, _ = _best_damage_ko(clone, getattr(battle, "opponent_active_pokemon", None), weather=weather)

    value = 0.0
    if boosted_kos >= 2:
        value = 20.0 + boosted_kos * 2.0 + min(6.0, meaningful_gain * 5.0)
    elif boosted_current_ko >= 1.0 and current_ko < 1.0:
        value = 9.0 + min(3.0, meaningful_gain * 3.0)

    return value, boosted_kos, after_move.id if after_move is not None else "", tuple(sorted(set(required_removals)))


def _find_primary_win_condition(battle: Any) -> tuple[Any | None, Any | None, StrategicPlan | None]:
    active = getattr(battle, "active_pokemon", None)
    opponent = getattr(battle, "opponent_active_pokemon", None)
    revealed = _revealed_opponents(battle)
    alive = _team_alive(battle)
    best: tuple[float, Any, Any, int, tuple[str, ...]] | None = None

    for candidate in alive:
        setup_moves = [m for m in _ordered_moves(candidate)[:4] if _move_id(m) in _SETUP_MOVES]
        for setup_move in setup_moves:
            value, ko_count, after_move, required = _candidate_sweep_value(battle, candidate, setup_move)
            if value <= 0:
                continue
            # Prefer healthy active candidates, then candidates that create more KOs.
            active_bonus = 2.0 if candidate is active else 0.0
            score = value + active_bonus + _hp_fraction(candidate)
            if best is None or score > best[0]:
                best = (score, candidate, setup_move, ko_count, required)

    spikes = 0
    for key, value in (getattr(battle, "opponent_side_conditions", {}) or {}).items():
        key_id = str(getattr(key, "name", key)).lower().replace(" ", "")
        if key_id == "spikes":
            try:
                spikes = int(value)
            except (TypeError, ValueError):
                spikes = 0
            break

    grounded_remaining = sum(1 for p in revealed if _is_grounded(p))
    hazard_goal = 0
    if any(_move_id(m) == "spikes" for p in alive for m in _ordered_moves(p)[:4]):
        if grounded_remaining >= 4:
            hazard_goal = 3
        elif grounded_remaining >= 2:
            hazard_goal = 2
        elif grounded_remaining >= 1:
            hazard_goal = 1

    cleanup_mode = False
    if len(revealed) <= 2 and active is not None and opponent is not None:
        best_ko, _, _ = _best_damage_ko(active, opponent, weather=_weather(battle))
        cleanup_mode = best_ko >= 1.0

    if best is None:
        return None, None, StrategicPlan(
            primary_win_condition=None,
            setup_move=None,
            sweep_ready=False,
            sweep_ko_count=0,
            hazard_goal=hazard_goal,
            hazard_layer=spikes,
            cleanup_mode=cleanup_mode,
            required_removals=(),
            reason="no high-confidence setup sweep window detected",
        )

    _, candidate, setup_move, ko_count, required = best
    return candidate, setup_move, StrategicPlan(
        primary_win_condition=_species_key(candidate),
        setup_move=_move_id(setup_move),
        sweep_ready=ko_count >= 2,
        sweep_ko_count=ko_count,
        hazard_goal=hazard_goal,
        hazard_layer=spikes,
        cleanup_mode=cleanup_mode,
        required_removals=required,
        reason=(
            f"primary win condition: {_species_key(candidate)} via {_move_id(setup_move)}; "
            f"one setup turn creates {ko_count} boosted KO window(s)"
        ),
    )


def _available_move_map(battle: Any, active: Any) -> dict[int, Any]:
    moves = _ordered_moves(active)[:4]
    available_ids = {_move_id(m) for m in (getattr(battle, "available_moves", []) or [])}
    return {
        int(action): moves[int(action)]
        for action in range(min(4, len(moves)))
        if _move_id(moves[int(action)]) in available_ids
    }


def strategic_opportunity_override(
    battle: Any,
    legal_actions: list[int],
    model_action: int,
) -> tuple[int | None, str]:
    """Convert high-confidence strategic opportunities into legal actions.

    This layer is intentionally narrower than a full search engine. It uses
    only revealed information and concrete damage estimates, but it maintains
    a board-level concept of a primary win condition, hazard target, and
    cleanup state instead of evaluating setup one turn in isolation.
    """
    active = getattr(battle, "active_pokemon", None)
    opponent = getattr(battle, "opponent_active_pokemon", None)
    if active is None or opponent is None or getattr(battle, "force_switch", False):
        return None, ""

    candidate, setup_move, plan = _find_primary_win_condition(battle)
    if plan is None:
        return None, ""

    move_map = _available_move_map(battle, active)
    weather = _weather(battle)

    # Never manufacture a setup/hazard turn from a compromised active.
    if _hp_fraction(active) < 0.55:
        # A clean primary win condition can still be activated by switching into it.
        if candidate is not None and candidate is not active and plan.sweep_ready:
            for action in legal_actions:
                if action < 4:
                    continue
                slots = _switch_slots(battle)
                idx = int(action) - 4
                if not (0 <= idx < len(slots)) or slots[idx] is not candidate:
                    continue
                safe, incoming_pct, incoming_note = _incoming_safe(battle, candidate, opponent)
                if safe:
                    return int(action), (
                        f"win-condition activation: switch to {plan.primary_win_condition} "
                        f"so the identified {plan.setup_move} sweep plan stays available; "
                        f"worst revealed incoming damage {incoming_pct:.1f}% ({incoming_note})"
                    )
        return None, ""

    # Cleanup takes priority over extra setup. Never set up when an immediate
    # reliable KO ends the current interaction.
    current_ko, current_idx, current_move = _best_damage_ko(active, opponent, weather=weather)
    if plan.cleanup_mode and current_ko >= 1.0 and current_idx in move_map:
        return None, ""

    # --- Existing setup state: execute the sweep rather than resetting the plan.
    if candidate is active and setup_move is not None:
        boosts = getattr(active, "boosts", {}) or {}
        has_positive_setup = any(int(v or 0) > 0 for stat, v in boosts.items() if stat in {"atk", "spa", "spe"})
        best_after_ko, best_after_idx, best_after_move = _best_damage_ko(active, opponent, weather=weather)
        if has_positive_setup and best_after_ko >= 1.0 and best_after_idx in move_map:
            if model_action < 4:
                selected = move_map.get(int(model_action))
                if selected is None or _base_power(selected) <= 0 or _move_id(selected) != _move_id(best_after_move):
                    return best_after_idx, (
                        f"cleanup execution: {plan.primary_win_condition} is already boosted and "
                        f"{_move_id(best_after_move)} is a guaranteed KO; execute the sweep"
                    )

    # --- Spikes plan -----------------------------------------------------
    spikes = plan.hazard_layer
    spikes_idx = next((idx for idx, move in move_map.items() if _move_id(move) == "spikes"), None)
    if spikes_idx is not None and spikes < min(3, max(1, plan.hazard_goal)):
        grounded_remaining = [p for p in _revealed_opponents(battle) if _is_grounded(p)]
        opponent_move_ids = {_move_id(m) for m in (getattr(opponent, "moves", {}) or {}).values()}
        has_spinner = "rapidspin" in opponent_move_ids or any(_has_revealed_move(p, {"rapidspin"}) for p in _revealed_opponents(battle))
        our_ghost = any(
            "ghost" in {str(getattr(t, "name", t)).lower() for t in (getattr(p, "types", ()) or ())}
            for p in _team_alive(battle)
        )
        safe, incoming_pct, incoming_note = _incoming_safe(battle, active, opponent)
        if len(grounded_remaining) >= 2 and safe and current_ko < 1.0:
            # If the opponent has a known spinner and no spinblocker, don't waste
            # the third layer automatically; layers 1-2 remain useful if safe.
            if spikes >= 2 and has_spinner and not our_ghost:
                return None, ""
            reason = (
                f"hazard plan: add Spikes layer {spikes + 1}/3; "
                f"{len(grounded_remaining)} grounded opponent(s) remain"
            )
            if has_spinner and our_ghost:
                reason += "; preserve a spinblocker so the stack has lasting value"
            if incoming_pct is not None:
                reason += f"; worst revealed damage {incoming_pct:.1f}% ({incoming_note})"
            return spikes_idx, reason

    # --- Primary sweep setup --------------------------------------------
    if candidate is active and setup_move is not None:
        setup_idx = next((idx for idx, move in move_map.items() if _move_id(move) == _move_id(setup_move)), None)
        if setup_idx is not None:
            safe, incoming_pct, incoming_note = _incoming_safe(battle, active, opponent)
            value, ko_count, _, required = _candidate_sweep_value(battle, active, move_map[setup_idx])
            if safe and value > 0:
                # A guaranteed current KO beats a setup turn.
                if current_ko < 1.0:
                    return setup_idx, (
                        f"{plan.reason}; setup now with {_move_id(move_map[setup_idx])}; "
                        f"required removals: {', '.join(required) if required else 'none'}; "
                        f"worst revealed incoming damage {incoming_pct:.1f}% ({incoming_note})"
                    )

    # --- Activate an inactive primary win condition when the switch itself
    # creates a real, concrete setup opportunity.
    if candidate is not None and candidate is not active and plan.sweep_ready and model_action >= 4:
        slots = _switch_slots(battle)
        idx = int(model_action) - 4
        selected_target = slots[idx] if 0 <= idx < len(slots) else None
        if selected_target is not candidate:
            safe, incoming_pct, incoming_note = _incoming_safe(battle, candidate, opponent)
            if safe:
                return_action = next(
                    (int(action) for action in legal_actions if action >= 4 and
                     0 <= int(action) - 4 < len(slots) and slots[int(action) - 4] is candidate),
                    None,
                )
                if return_action is not None:
                    return return_action, (
                        f"win-condition activation: prefer {plan.primary_win_condition} over the model's switch; "
                        f"it is the identified {plan.setup_move} cleaner with {plan.sweep_ko_count} boosted KO window(s); "
                        f"worst revealed incoming damage {incoming_pct:.1f}% ({incoming_note})"
                    )

    return None, ""
