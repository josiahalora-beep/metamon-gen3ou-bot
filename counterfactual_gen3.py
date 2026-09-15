from __future__ import annotations

import argparse
import json
import random
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Iterable

from battle_ai.damage import calculate_damage, type_multiplier
from battle_ai.opponent_model import OpponentModel, PredictedResponse
from metamon.interface import consistent_move_order, consistent_pokemon_order

TEAM_MATRIX_NAMES = (
    "01_tss_balance", "02_aerobi_zapdos", "03_four_bands", "04_curselax_forredol",
    "05_blue_offense", "06_refresh_mixmence", "08_mixzap_houdini", "09_jynx_special_offense",
    "10_raikou_slop", "11_elite_big5_tss", "12_big5_double_band", "13_zapdug_balance",
    "14_cunedol_control", "15_triple_natural_cure", "16_ddtar_starmie_spikes",
    "17_jolteon_spike_stack", "18_defensive_tss_control", "19_jynx_special_offense_elite",
    "20_magneton_physical_offense",
)

KEY_WIN_CONDITION = {
    "tyranitar", "salamence", "metagross", "snorlax", "celebi", "suicune",
    "zapdos", "starmie", "jolteon", "heracross", "dugtrio",
}
STATUS_PENALTY = {"tox": 14.0, "psn": 9.0, "brn": 8.0, "par": 8.0, "slp": 10.0, "frz": 25.0}
SELF_KO = {"explosion", "selfdestruct", "memento"}


@dataclass(frozen=True)
class ResponseBranch:
    kind: str
    probability: float
    target: str = ""
    move: str = ""


@dataclass(frozen=True)
class CounterfactualResult:
    matrix: str
    battle_id: str
    turn: int
    base_action: int
    predictive_action: int
    expected_delta: float
    worst_likely_delta: float
    false_prediction: bool
    win_condition_sacrifice: bool
    self_ko_override: bool
    base_value: float
    predictive_value: float
    top_response: str


def _key(value: Any) -> str:
    raw = getattr(value, "id", getattr(value, "name", value))
    return str(raw or "").lower().replace(" ", "").replace("-", "").replace("_", "")


def _species(value: Any) -> str:
    raw = getattr(value, "species", getattr(value, "name", value))
    return str(raw or "").lower().replace(" ", "").replace("-", "").replace("_", "")


def _hp(pokemon: Any) -> float:
    try:
        return min(1.0, max(0.0, float(getattr(pokemon, "current_hp_fraction", 1.0))))
    except (TypeError, ValueError):
        return 1.0


def _move_for_action(battle: Any, action: int) -> Any | None:
    if not 0 <= action < 4:
        return None
    moves = list((getattr(battle.active_pokemon, "moves", {}) or {}).values())
    try:
        moves = list(consistent_move_order(moves))
    except ValueError:
        moves.sort(key=_key)
    return moves[action] if action < len(moves) else None


def _switch_for_action(battle: Any, action: int) -> Any | None:
    if action < 4:
        return None
    slots = [
        p for p in (getattr(battle, "team", {}) or {}).values()
        if p is not None and not getattr(p, "fainted", False) and not getattr(p, "active", False)
    ]
    try:
        slots = list(consistent_pokemon_order(slots))
    except ValueError:
        slots.sort(key=_species)
    idx = action - 4
    return slots[idx] if 0 <= idx < len(slots) else None


def _legal_actions(battle: Any) -> list[int]:
    moves = list((getattr(battle.active_pokemon, "moves", {}) or {}).values())
    switches = [
        p for p in (getattr(battle, "team", {}) or {}).values()
        if p is not None and not getattr(p, "fainted", False) and not getattr(p, "active", False)
    ]
    try:
        moves = list(consistent_move_order(moves))
        switches = list(consistent_pokemon_order(switches))
    except ValueError:
        moves.sort(key=_key)
        switches.sort(key=_species)
    return list(range(len(moves))) + [4 + i for i in range(len(switches))]


def _remaining_team_value(team: Iterable[Any]) -> float:
    value = 0.0
    for pokemon in team:
        hp = _hp(pokemon)
        if getattr(pokemon, "fainted", False) or hp <= 0:
            value -= 40.0
            continue
        value += 26.0 * hp + 4.0
        value -= STATUS_PENALTY.get(_key(getattr(pokemon, "status", "")), 0.0)
        boosts = getattr(pokemon, "boosts", {}) or {}
        value += 2.0 * max(0, int(boosts.get("atk", 0) or 0))
        value += 2.0 * max(0, int(boosts.get("spa", 0) or 0))
        value += 1.0 * max(0, int(boosts.get("spe", 0) or 0))
    return value


def _coverage_value(our_team: Iterable[Any], target: Any) -> float:
    if target is None:
        return 0.0
    target_types = list(getattr(target, "types", ()) or ())
    score = 0.0
    for pokemon in our_team:
        if getattr(pokemon, "fainted", False):
            continue
        for move in (getattr(pokemon, "moves", {}) or {}).values():
            if int(getattr(move, "base_power", 0) or 0) <= 0:
                continue
            mult = type_multiplier(_key(getattr(move, "type", "")), target_types)
            score += 1.0 if mult >= 2.0 else 0.5 if mult > 1.0 else 0.0
    return min(8.0, score)


def position_value(our_team: Iterable[Any], opponent_team: Iterable[Any], our_active: Any, opponent_active: Any) -> float:
    ours = _remaining_team_value(our_team)
    theirs = _remaining_team_value(opponent_team)
    our_speed = float((getattr(our_active, "stats", {}) or {}).get("spe", 0) or 0)
    opp_speed = float((getattr(opponent_active, "stats", {}) or {}).get("spe", 0) or 0)
    speed = 3.0 if our_speed > opp_speed else -3.0 if our_speed < opp_speed else 0.0
    coverage = _coverage_value(our_team, opponent_active) - _coverage_value(opponent_team, our_active)
    return ours - theirs + speed + coverage


def _copy_pokemon(pokemon: Any, *, active: bool | None = None, hp_fraction: float | None = None) -> Any:
    clone = NS(**vars(pokemon))
    if active is not None:
        clone.active = active
    if hp_fraction is not None:
        clone.current_hp_fraction = hp_fraction
        max_hp = float(getattr(clone, "max_hp", 0) or 0)
        if max_hp > 0:
            clone.current_hp = max(0, int(round(max_hp * hp_fraction)))
        if hp_fraction <= 0.0:
            clone.fainted = True
    return clone


def _damage_fraction(attacker: Any, defender: Any, move: Any) -> tuple[float, bool, float]:
    result = calculate_damage(attacker, defender, move, weather="")
    if not result.reliable:
        return 0.0, False, 0.0
    hp = max(1.0, float(getattr(defender, "current_hp", 1) or 1))
    return min(1.0, float(result.max_damage) / hp), True, float(result.ko_probability)


def _responses(battle: Any, model: OpponentModel) -> list[ResponseBranch]:
    raw = model.predict_responses(battle)
    branches = [ResponseBranch(r.kind, r.probability, r.target, r.move) for r in raw if r.kind != "unknown" and r.probability > 0.01]
    total = sum(max(0.0, b.probability) for b in branches)
    if not branches or total <= 0:
        return [ResponseBranch("unknown", 1.0)]
    if abs(total - 1.0) > 1e-9:
        branches = [ResponseBranch(b.kind, b.probability / total, b.target, b.move) for b in branches]
    return branches


def _response_target(battle: Any, response: ResponseBranch) -> Any | None:
    if response.kind != "switch" or not response.target:
        return None
    return next(
        (p for p in (getattr(battle, "opponent_team", {}) or {}).values()
         if p is not None and not getattr(p, "fainted", False) and _species(p) == response.target),
        None,
    )


def _apply_turn(battle: Any, action: int, response: ResponseBranch) -> float:
    our_team = list((getattr(battle, "team", {}) or {}).values())
    opp_team = list((getattr(battle, "opponent_team", {}) or {}).values())
    our_active = getattr(battle, "active_pokemon", None)
    opp_active = getattr(battle, "opponent_active_pokemon", None)
    if our_active is None or opp_active is None:
        return position_value(our_team, opp_team, our_active, opp_active)

    new_our = {id(p): _copy_pokemon(p) for p in our_team}
    new_opp = {id(p): _copy_pokemon(p) for p in opp_team}
    our_active2 = new_our.get(id(our_active), _copy_pokemon(our_active, active=True))
    opp_active2 = new_opp.get(id(opp_active), _copy_pokemon(opp_active, active=True))

    switched = _switch_for_action(battle, action)
    if switched is not None:
        our_active2 = new_our.get(id(switched), _copy_pokemon(switched, active=True))
    else:
        move = _move_for_action(battle, action)
        target = opp_active2
        predicted_switch = _response_target(battle, response)
        if predicted_switch is not None:
            target = new_opp.get(id(predicted_switch), target)
            target.active = True
        if move is not None and int(getattr(move, "base_power", 0) or 0) > 0:
            dmg, reliable, ko = _damage_fraction(our_active2, target, move)
            if reliable:
                target_hp = _hp(target)
                target.current_hp_fraction = max(0.0, target_hp * (1.0 - dmg))
                if ko >= 0.95 or target.current_hp_fraction <= 0.01:
                    target.current_hp_fraction = 0.0
                    target.current_hp = 0
                    target.fainted = True
        opp_active2 = target

    if response.kind == "attack" and response.move:
        opp_move = next(
            (m for m in (getattr(opp_active2, "moves", {}) or {}).values() if _key(m) == response.move),
            None,
        )
        if opp_move is not None and int(getattr(opp_move, "base_power", 0) or 0) > 0 and not getattr(our_active2, "fainted", False):
            dmg, reliable, ko = _damage_fraction(opp_active2, our_active2, opp_move)
            if reliable:
                our_active2.current_hp_fraction = max(0.0, _hp(our_active2) * (1.0 - dmg))
                if ko >= 0.95 or our_active2.current_hp_fraction <= 0.01:
                    our_active2.current_hp_fraction = 0.0
                    our_active2.current_hp = 0
                    our_active2.fainted = True
    elif response.kind == "setup":
        boosts = dict(getattr(opp_active2, "boosts", {}) or {})
        boosts["atk"] = min(6, int(boosts.get("atk", 0) or 0) + 1)
        boosts["spa"] = min(6, int(boosts.get("spa", 0) or 0) + 1)
        opp_active2.boosts = boosts
    elif response.kind == "passive":
        if not _key(getattr(our_active2, "status", "")):
            our_active2.status = "tox"

    return position_value(new_our.values(), new_opp.values(), our_active2, opp_active2)


def evaluate_actions(battle: Any, base_action: int, predictive_action: int, *, matrix: str, model: OpponentModel | None = None) -> CounterfactualResult:
    model = model or OpponentModel()
    branches = _responses(battle, model)
    base_values = [_apply_turn(battle, base_action, b) for b in branches]
    predictive_values = [_apply_turn(battle, predictive_action, b) for b in branches]
    base_expected = sum(b.probability * value for b, value in zip(branches, base_values))
    predictive_expected = sum(b.probability * value for b, value in zip(branches, predictive_values))
    expected_delta = predictive_expected - base_expected

    likely = sorted(
        zip(branches, base_values, predictive_values),
        key=lambda row: row[0].probability,
        reverse=True,
    )
    branch_deltas = [predictive_value - base_value for _, base_value, predictive_value in likely[:3]]
    worst_likely_delta = min(branch_deltas) if branch_deltas else expected_delta

    predictive_move = _move_for_action(battle, predictive_action)
    self_ko_override = bool(predictive_move is not None and _key(predictive_move) in SELF_KO and base_action >= 4)
    active = getattr(battle, "active_pokemon", None)
    active_species = _species(active) if active is not None else ""
    is_key = active_species in KEY_WIN_CONDITION
    low_hp = _hp(active) <= 0.40 if active is not None else False
    win_condition_sacrifice = bool(base_action >= 4 and predictive_action < 4 and (is_key or low_hp) and expected_delta < -2.0)

    top = branches[0]
    return CounterfactualResult(
        matrix=matrix,
        battle_id=str(getattr(battle, "battle_tag", "")),
        turn=int(getattr(battle, "turn", 0) or 0),
        base_action=int(base_action),
        predictive_action=int(predictive_action),
        expected_delta=expected_delta,
        worst_likely_delta=worst_likely_delta,
        false_prediction=bool(top.kind == "switch" and expected_delta < -2.0),
        win_condition_sacrifice=win_condition_sacrifice,
        self_ko_override=self_ko_override,
        base_value=base_expected,
        predictive_value=predictive_expected,
        top_response=f"{top.kind}:{top.target or top.move or '-'} {top.probability:.0%}",
    )


def _synthetic_battle(rng: random.Random, index: int, matrix: str) -> Any:
    move_data = {
        "earthquake": (100, "Ground"), "rockslide": (75, "Rock"), "doubleedge": (100, "Normal"),
        "dragondance": (0, "Dragon"), "explosion": (250, "Normal"), "protect": (0, "Normal"),
        "surf": (95, "Water"), "icebeam": (95, "Ice"), "calmmind": (0, "Psychic"), "recover": (0, "Normal"),
        "thunderbolt": (95, "Electric"), "toxic": (0, "Poison"), "meteor_mash": (100, "Steel"),
        "spikes": (0, "Ground"), "drillpeck": (80, "Flying"), "leechseed": (0, "Grass"),
        "psychic": (90, "Psychic"), "gigadrain": (75, "Grass"), "roar": (0, "Normal"),
    }
    archetypes = [
        ("tyranitar", ("Rock", "Dark"), ["rockslide", "earthquake", "dragondance", "doubleedge"]),
        ("metagross", ("Steel", "Psychic"), ["meteor_mash", "earthquake", "explosion", "protect"]),
        ("swampert", ("Water", "Ground"), ["surf", "icebeam", "earthquake", "roar"]),
        ("skarmory", ("Steel", "Flying"), ["spikes", "drillpeck", "toxic", "protect"]),
        ("celebi", ("Psychic", "Grass"), ["psychic", "gigadrain", "leechseed", "calmmind"]),
        ("salamence", ("Dragon", "Flying"), ["dragondance", "earthquake", "rockslide", "meteor_mash"]),
        ("suicune", ("Water",), ["surf", "icebeam", "calmmind", "roar"]),
        ("zapdos", ("Electric", "Flying"), ["thunderbolt", "drillpeck", "roar", "recover"]),
        ("heracross", ("Bug", "Fighting"), ["brickbreak", "rockslide", "swordsdance", "megahorn"]),
        ("snorlax", ("Normal",), ["doubleedge", "earthquake", "curse", "recover"]),
    ]
    def mon(name: str, types: tuple[str, ...], ids: list[str], hp: float, active: bool = False) -> Any:
        moves = {
            move_id: NS(id=move_id, name=move_id, base_power=move_data[move_id][0], type=NS(name=move_data[move_id][1]), category=NS(name="Physical"), priority=0)
            for move_id in ids if move_id in move_data
        }
        stats = {"atk": 236, "def": 236, "spa": 236, "spd": 236, "spe": rng.choice([180, 220, 260, 300, 340])}
        return NS(
            name=name, species=name, base_species=name, types=types,
            current_hp=max(1, int(300 * hp)), max_hp=300, current_hp_fraction=hp,
            status=rng.choice(["", "", "", "brn", "tox", "par"]), item="Leftovers", ability="",
            level=100, base_stats={"hp": 100, "atk": 120, "def": 100, "spa": 100, "spd": 100, "spe": 100},
            stats=stats, boosts={"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
            moves=moves, fainted=False, active=active,
        )
    own_name, own_types, own_moves = rng.choice(archetypes)
    opp_name, opp_types, opp_moves = rng.choice([x for x in archetypes if x[0] != own_name])
    own = mon(own_name, own_types, own_moves, rng.choice([0.15, 0.25, 0.35, 0.50, 0.70, 1.0]), True)
    opp = mon(opp_name, opp_types, opp_moves, rng.choice([0.15, 0.30, 0.50, 0.70, 1.0]), True)
    teammates = [mon(*rng.choice(archetypes), rng.choice([0.25, 0.50, 0.75, 1.0])) for _ in range(3)]
    for p in teammates:
        p.active = False
    team = {f"{p.name}_{i}": p for i, p in enumerate([own] + teammates)}
    opponent_team = {f"{p.name}_{i}": p for i, p in enumerate([opp])}
    return NS(
        battle_tag=f"synthetic-{matrix}-{index}", turn=rng.randint(2, 80), format="gen3ou",
        player_username="synthetic", opponent_username="synthetic", active_pokemon=own,
        opponent_active_pokemon=opp, available_moves=list(own.moves.values()),
        available_switches=teammates, team=team, opponent_team=opponent_team,
        force_switch=False, reviving=False, can_tera=None, weather={},
        side_conditions={}, opponent_side_conditions={}, fields={}
    )


def generate_synthetic_cases(samples: int, seed: int = 7) -> list[tuple[str, Any, int, int]]:
    rng = random.Random(seed)
    cases = []
    attempts = max(0, int(samples))
    for i in range(attempts):
        matrix = TEAM_MATRIX_NAMES[i % len(TEAM_MATRIX_NAMES)]
        battle = _synthetic_battle(rng, i, matrix)
        legal = _legal_actions(battle)
        if len(legal) < 2:
            continue
        base = rng.choice(legal)
        predictive = rng.choice(legal)
        if base == predictive:
            predictive = legal[(legal.index(base) + 1) % len(legal)]
        cases.append((matrix, battle, base, predictive))
    return cases


def run_synthetic(samples: int, seed: int = 7) -> dict[str, Any]:
    results = [evaluate_actions(battle, base, predictive, matrix=matrix) for matrix, battle, base, predictive in generate_synthetic_cases(samples, seed)]
    return {
        "mode": "synthetic", "seed": seed, "requested_samples": int(samples), "evaluated_cases": len(results),
        "negative_delta_cases": sum(r.expected_delta < -2.0 for r in results),
        "false_prediction_cases": sum(r.false_prediction for r in results),
        "win_condition_sacrifice_cases": sum(r.win_condition_sacrifice for r in results),
        "self_ko_overrides": sum(r.self_ko_override for r in results),
        "mean_delta": sum(r.expected_delta for r in results) / len(results) if results else 0.0,
        "worst_delta": min((r.expected_delta for r in results), default=0.0),
        "by_matrix": {
            matrix: {
                "cases": len(rows),
                "negative_delta_rate": sum(r.expected_delta < -2.0 for r in rows) / len(rows),
                "mean_delta": sum(r.expected_delta for r in rows) / len(rows),
                "false_prediction_rate": sum(r.false_prediction for r in rows) / len(rows),
                "sacrifice_rate": sum(r.win_condition_sacrifice for r in rows) / len(rows),
            }
            for matrix in TEAM_MATRIX_NAMES
            if (rows := [r for r in results if r.matrix == matrix])
        },
        "worst_cases": [asdict(r) for r in sorted(results, key=lambda r: (r.expected_delta, r.worst_likely_delta))[:50]],
    }


def audit_recorded(database: Path, limit: int = 0) -> list[CounterfactualResult]:
    db = sqlite3.connect(database)
    try:
        sql = "SELECT battle_id,turn,snapshot_json,chosen_action FROM turns ORDER BY id"
        rows = db.execute(sql + (" LIMIT ?" if limit > 0 else ""), (limit,) if limit > 0 else ()).fetchall()
    finally:
        db.close()
    from benchmark_counterfactual import battle_from_snapshot
    results = []
    evaluator_model = OpponentModel()
    from battle_ai.evaluator import TacticalEvaluator
    for battle_id, turn, raw, base_action in rows:
        try:
            battle = battle_from_snapshot(json.loads(raw))
            legal = _legal_actions(battle)
            if int(base_action) not in legal:
                continue
            final, _ = TacticalEvaluator().evaluate(battle, legal, int(base_action))
            if int(final) == int(base_action):
                continue
            results.append(evaluate_actions(battle, int(base_action), int(final), matrix="recorded", model=evaluator_model))
        except Exception:
            continue
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Gen 3 OU counterfactual audit")
    parser.add_argument("--synthetic", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--database", default="battle_data/battles.db")
    parser.add_argument("--recorded-limit", type=int, default=0)
    parser.add_argument("--output", default="battle_data/counterfactual_gen3_report.json")
    args = parser.parse_args()
    payload: dict[str, Any] = {}
    if args.synthetic > 0:
        payload["synthetic"] = run_synthetic(args.synthetic, args.seed)
    if args.database:
        try:
            recorded = audit_recorded(Path(args.database), args.recorded_limit)
            payload["recorded"] = {
                "evaluated_overrides": len(recorded),
                "negative_delta_cases": sum(r.expected_delta < -2.0 for r in recorded),
                "false_prediction_cases": sum(r.false_prediction for r in recorded),
                "win_condition_sacrifice_cases": sum(r.win_condition_sacrifice for r in recorded),
                "self_ko_overrides": sum(r.self_ko_override for r in recorded),
                "mean_delta": sum(r.expected_delta for r in recorded) / len(recorded) if recorded else 0.0,
                "worst_delta": min((r.expected_delta for r in recorded), default=0.0),
                "worst_cases": [asdict(r) for r in sorted(recorded, key=lambda r: (r.expected_delta, r.worst_likely_delta))[:100]],
            }
        except (sqlite3.Error, FileNotFoundError):
            payload["recorded"] = {"error": "recorded database unavailable"}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "synthetic": {k: payload["synthetic"][k] for k in ("evaluated_cases", "negative_delta_cases", "false_prediction_cases", "win_condition_sacrifice_cases", "self_ko_overrides", "mean_delta", "worst_delta")} if "synthetic" in payload else None,
        "recorded": {k: payload["recorded"].get(k) for k in ("evaluated_overrides", "negative_delta_cases", "false_prediction_cases", "win_condition_sacrifice_cases", "self_ko_overrides", "mean_delta", "worst_delta")} if "recorded" in payload else None,
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
