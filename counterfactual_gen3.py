from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Iterable

from battle_ai.damage import calculate_damage, type_multiplier
from battle_ai.opponent_model import OpponentModel, PredictedResponse
from benchmark_counterfactual import battle_from_snapshot


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
    return str(getattr(value, "id", getattr(value, "name", value)) or "").lower().replace(" ", "").replace("-", "").replace("_", "")


def _species(value: Any) -> str:
    return _key(getattr(value, "species", getattr(value, "name", value)))


def _move_for_action(battle: Any, action: int) -> Any | None:
    if action < 0 or action >= 4:
        return None
    moves = list((getattr(battle.active_pokemon, "moves", {}) or {}).values())
    by_id = {_key(m): m for m in moves}
    try:
        from metamon.interface import consistent_move_order
        moves = list(consistent_move_order(moves))
    except ValueError:
        moves = sorted(moves, key=_key)
    return moves[action] if action < len(moves) else None


def _switch_for_action(battle: Any, action: int) -> Any | None:
    if action < 4:
        return None
    slots = [p for p in (getattr(battle, "team", {}) or {}).values()
             if p is not None and not getattr(p, "fainted", False) and not getattr(p, "active", False)]
    try:
        from metamon.interface import consistent_pokemon_order
        slots = list(consistent_pokemon_order(slots))
    except ValueError:
        slots = sorted(slots, key=_species)
    idx = action - 4
    return slots[idx] if 0 <= idx < len(slots) else None


def _legal_actions(battle: Any) -> list[int]:
    moves = list((getattr(battle.active_pokemon, "moves", {}) or {}).values())
    try:
        from metamon.interface import consistent_move_order, consistent_pokemon_order
        moves = list(consistent_move_order(moves))
        switches = [p for p in (getattr(battle, "team", {}) or {}).values()
                    if p is not None and not getattr(p, "fainted", False) and not getattr(p, "active", False)]
        switches = list(consistent_pokemon_order(switches))
    except ValueError:
        moves = sorted(moves, key=_key)
        switches = sorted(
            [p for p in (getattr(battle, "team", {}) or {}).values()
             if p is not None and not getattr(p, "fainted", False) and not getattr(p, "active", False)],
            key=_species,
        )
    return list(range(len(moves))) + [4 + i for i in range(len(switches))]


def _hp(pokemon: Any) -> float:
    return min(1.0, max(0.0, float(getattr(pokemon, "current_hp_fraction", 1.0) or 1.0)))


def _status_penalty(pokemon: Any) -> float:
    return STATUS_PENALTY.get(_key(getattr(pokemon, "status", "")), 0.0)


def _remaining_team_value(team: Iterable[Any]) -> float:
    value = 0.0
    for pokemon in team:
        if pokemon is None:
            continue
        hp = _hp(pokemon)
        value += 22.0 * hp
        if getattr(pokemon, "fainted", False) or hp <= 0:
            value -= 35.0
        else:
            value += 4.0
        value -= _status_penalty(pokemon)
        boosts = getattr(pokemon, "boosts", {}) or {}
        value += 2.0 * max(0, int(boosts.get("atk", 0) or 0))
        value += 2.0 * max(0, int(boosts.get("spa", 0) or 0))
        value += 1.0 * max(0, int(boosts.get("spe", 0) or 0))
    return value


def _coverage_value(our_team: Iterable[Any], target: Any) -> float:
    if target is None:
        return 0.0
    target_types = list(getattr(target, "types", ()) or ())
    threats = 0.0
    for pokemon in our_team:
        if getattr(pokemon, "fainted", False):
            continue
        for move in (getattr(pokemon, "moves", {}) or {}).values():
            if int(getattr(move, "base_power", 0) or 0) <= 0:
                continue
            mult = type_multiplier(_key(getattr(move, "type", "")), target_types)
            if mult >= 2.0:
                threats += 1.0
            elif mult > 1.0:
                threats += 0.5
    return min(8.0, threats)


def position_value(our_team: Iterable[Any], opponent_team: Iterable[Any], our_active: Any, opponent_active: Any) -> float:
    our = _remaining_team_value(our_team)
    opp = _remaining_team_value(opponent_team)
    speed_us = float((getattr(our_active, "stats", {}) or {}).get("spe", 0) or 0)
    speed_them = float((getattr(opponent_active, "stats", {}) or {}).get("spe", 0) or 0)
    speed = 3.0 if speed_us > speed_them else -3.0 if speed_us < speed_them else 0.0
    coverage = _coverage_value(our_team, opponent_active) - _coverage_value(opponent_team, our_active)
    return (our - opp) + speed + coverage


def _copy_pokemon(pokemon: Any, *, active: bool | None = None, hp_fraction: float | None = None) -> Any:
    clone = NS(**vars(pokemon))
    if active is not None:
        clone.active = active
    if hp_fraction is not None:
        clone.current_hp_fraction = hp_fraction
        max_hp = float(getattr(clone, "max_hp", 0) or 0)
        if max_hp > 0:
            clone.current_hp = max(0, int(round(max_hp * hp_fraction)))
    return clone


def _damage_fraction(attacker: Any, defender: Any, move: Any, weather: str = "") -> tuple[float, bool, float]:
    result = calculate_damage(attacker, defender, move, weather=weather)
    if not result.reliable:
        return 0.0, False, 0.0
    hp = max(1.0, float(getattr(defender, "current_hp", 1) or 1))
    return min(1.0, float(result.max_damage) / hp), True, float(result.ko_probability)


def _response_target(battle: Any, response: PredictedResponse) -> Any | None:
    if response.kind != "switch" or not response.target:
        return None
    for pokemon in (getattr(battle, "opponent_team", {}) or {}).values():
        if not getattr(pokemon, "fainted", False) and _species(pokemon) == response.target:
            return pokemon
    return None


def _responses(battle: Any, model: OpponentModel) -> list[ResponseBranch]:
    raw = model.predict_responses(battle)
    return [ResponseBranch(r.kind, r.probability, r.target, r.move) for r in raw if r.kind != "unknown" and r.probability > 0.01]


def _apply_turn(battle: Any, action: int, response: ResponseBranch) -> float:
    our_active = getattr(battle, "active_pokemon", None)
    opp_active = getattr(battle, "opponent_active_pokemon", None)
    if our_active is None or opp_active is None:
        return position_value(getattr(battle, "team", {}).values(), getattr(battle, "opponent_team", {}).values(), our_active, opp_active)

    our_team = list((getattr(battle, "team", {}) or {}).values())
    opp_team = list((getattr(battle, "opponent_team", {}) or {}).values())
    new_our = {id(p): _copy_pokemon(p) for p in our_team}
    new_opp = {id(p): _copy_pokemon(p) for p in opp_team}
    our_active2 = new_our.get(id(our_active), _copy_pokemon(our_active, active=True))
    opp_active2 = new_opp.get(id(opp_active), _copy_pokemon(opp_active, active=True))

    selected_switch = _switch_for_action(battle, action)
    if selected_switch is not None:
        our_active2 = new_our.get(id(selected_switch), _copy_pokemon(selected_switch, active=True))
    else:
        move = _move_for_action(battle, action)
        response_target = _response_target(battle, PredictedResponse(response.kind, response.probability, response.target, response.move))
        target2 = new_opp.get(id(response_target), opp_active2) if response_target is not None else opp_active2
        if move is not None and int(getattr(move, "base_power", 0) or 0) > 0:
            dmg, reliable, ko = _damage_fraction(our_active2, target2, move, weather="")
            if reliable:
                target2.current_hp_fraction = max(0.0, _hp(target2) * (1.0 - dmg))
                if ko >= 0.95 or target2.current_hp_fraction <= 0.01:
                    target2.current_hp_fraction = 0.0
                    target2.current_hp = 0
                    target2.fainted = True
        opp_active2 = target2

    if response.kind == "attack" and response.move:
        opp_move = None
        for move in (getattr(opp_active2, "moves", {}) or {}).values():
            if _key(move) == response.move:
                opp_move = move
                break
        if opp_move is not None and int(getattr(opp_move, "base_power", 0) or 0) > 0:
            dmg, reliable, ko = _damage_fraction(opp_active2, our_active2, opp_move, weather="")
            if reliable:
                our_active2.current_hp_fraction = max(0.0, _hp(our_active2) * (1.0 - dmg))
                if ko >= 0.95 or our_active2.current_hp_fraction <= 0.01:
                    our_active2.current_hp_fraction = 0.0
                    our_active2.current_hp = 0
                    our_active2.fainted = True
    elif response.kind == "setup":
        boosts = dict(getattr(opp_active2, "boosts", {}) or {})
        boosts["atk"] = int(boosts.get("atk", 0) or 0) + 1
        boosts["spa"] = int(boosts.get("spa", 0) or 0) + 1
        opp_active2.boosts = boosts
    elif response.kind == "passive":
        our_active2.status = getattr(our_active2, "status", "") or "tox"
    elif response.kind == "protect":
        pass

    for original in our_team:
        new = new_our[id(original)]
        if original is our_active:
            new.active = our_active2 is new
    for original in opp_team:
        new = new_opp[id(original)]
        if original is opp_active:
            new.active = opp_active2 is new
    new_our[id(our_active)] = our_active2
    new_opp[id(opp_active)] = opp_active2
    return position_value(new_our.values(), new_opp.values(), our_active2, opp_active2)


def evaluate_actions(battle: Any, base_action: int, predictive_action: int, *, matrix: str, model: OpponentModel | None = None) -> CounterfactualResult:
    model = model or OpponentModel()
    branches = _responses(battle, model)
    if not branches:
        branches = [ResponseBranch("unknown", 1.0)]
    base_values = [(b.probability, _apply_turn(battle, base_action, b)) for b in branches]
    predictive_values = [(b.probability, _apply_turn(battle, predictive_action, b)) for b in branches]
    base_expected = sum(p * v for p, v in base_values)
    predictive_expected = sum(p * v for p, v in predictive_values)
    expected_delta = predictive_expected - base_expected

    likely = sorted(zip(branches, base_values, predictive_values), key=lambda x: x[0].probability, reverse=True)
    worst_likely_delta = min(pv[2] - bv[1] for _, bv, pv in likely[:3]) if likely else expected_delta

    predictive_move = _move_for_action(battle, predictive_action)
    self_ko = predictive_move is not None and _key(predictive_move) in {"explosion", "selfdestruct", "memento"}
    active = getattr(battle, "active_pokemon", None)
    win_condition = _species(active) in KEY_WIN_CONDITION if active is not None else False
    sacrifice = base_action >= 4 and predictive_action < 4 and (win_condition or _hp(active) <= 0.40 if active is not None else False)

    top = branches[0] if branches else ResponseBranch("unknown", 1.0)
    return CounterfactualResult(
        matrix=matrix,
        battle_id=str(getattr(battle, "battle_tag", "")),
        turn=int(getattr(battle, "turn", 0) or 0),
        base_action=int(base_action), predictive_action=int(predictive_action),
        expected_delta=expected_delta, worst_likely_delta=worst_likely_delta,
        false_prediction=(top.kind == "switch" and expected_delta < -2.0),
        win_condition_sacrifice=sacrifice and expected_delta < -2.0,
        self_ko_override=self_ko and base_action >= 4,
        base_value=base_expected, predictive_value=predictive_expected,
        top_response=f"{top.kind}:{top.target or top.move or '-'} {top.probability:.0%}",
    )


def audit_recorded(database: Path, limit: int = 0, matrix_cycle: bool = True) -> list[CounterfactualResult]:
    import sqlite3
    db = sqlite3.connect(database)
    try:
        sql = "SELECT battle_id,turn,snapshot_json,chosen_action FROM turns ORDER BY id"
        params: tuple[Any, ...] = ()
        if limit > 0:
            sql += " LIMIT ?"
            params = (limit,)
        rows = db.execute(sql, params).fetchall()
    finally:
        db.close()

    results: list[CounterfactualResult] = []
    model = OpponentModel()
    for idx, (battle_id, turn, raw, base_action) in enumerate(rows):
        snapshot = json.loads(raw)
        battle = battle_from_snapshot(snapshot)
        legal = _legal_actions(battle)
        try:
            from battle_ai.evaluator import TacticalEvaluator
            final_action, _ = TacticalEvaluator().evaluate(battle, legal, int(base_action))
        except Exception:
            continue
        if int(final_action) == int(base_action):
            continue
        matrix = TEAM_MATRIX_NAMES[idx % len(TEAM_MATRIX_NAMES)] if matrix_cycle else "recorded"
        results.append(evaluate_actions(battle, int(base_action), int(final_action), matrix=matrix, model=model))
    return results


def generate_synthetic_cases(samples: int, seed: int = 7) -> list[tuple[str, Any, int, int]]:
    """Generate adversarial Gen 3 cases from the real damage/evaluator interfaces.

    Synthetic cases are explicitly marked as such. They do not claim to be the
    exact private set files; recorded-state mode is the authoritative audit for
    actual battles.
    """
    rng = random.Random(seed)
    species = [
        ("tyranitar", ("Rock", "Dark"), ["rockslide", "earthquake", "dragondance", "doubleedge"]),
        ("metagross", ("Steel", "Psychic"), ["meteormash", "earthquake", "explosion", "protect"]),
        ("swampert", ("Water", "Ground"), ["surf", "icebeam", "earthquake", "roar"]),
        ("skarmory", ("Steel", "Flying"), ["spikes", "drillpeck", "toxic", "protect"]),
        ("celebi", ("Psychic", "Grass"), ["psychic", "gigadrain", "leechseed", "calmmind"]),
        ("salamence", ("Dragon", "Flying"), ["dragondance", "earthquake", "rockslide", "fireblast"]),
        ("suicune", ("Water",), ["surf", "icebeam", "calmmind", "roar"]),
        ("zapdos", ("Electric", "Flying"), ["thunderbolt", "drillpeck", "roar", "rest"]),
        ("heracross", ("Bug", "Fighting"), ["megahorn", "brickbreak", "swordsdance", "rockslide"]),
        ("snorlax", ("Normal",), ["bodyslam", "curse", "earthquake", "rest"]),
    ]
    move_data = {
        "rockslide": (75, "Rock"), "earthquake": (100, "Ground"), "dragondance": (0, "Dragon"),
        "doubleedge": (100, "Normal"), "meteormash": (100, "Steel"), "explosion": (250, "Normal"),
        "protect": (0, "Normal"), "surf": (95, "Water"), "icebeam": (95, "Ice"), "roar": (0, "Normal"),
        "spikes": (0, "Ground"), "drillpeck": (80, "Flying"), "toxic": (0, "Poison"), "psychic": (90, "Psychic"),
        "gigadrain": (75, "Grass"), "leechseed": (0, "Grass"), "calmmind": (0, "Psychic"), "fireblast": (120, "Fire"),
        "thunderbolt": (95, "Electric"), "rest": (0, "Psychic"), "megahorn": (120, "Bug"), "brickbreak": (75, "Fighting"),
        "swordsdance": (0, "Normal"), "bodyslam": (85, "Normal"), "curse": (0, "Ghost"),
    }

    def mon(name: str, types: tuple[str, ...], move_ids: list[str], hp: float, active=False):
        moves = {}
        for move_id in move_ids:
            power, typ = move_data[move_id]
            moves[move_id] = NS(id=move_id, name=move_id, base_power=power, type=NS(name=typ), category=NS(name="Physical"), priority=0)
        base = {"hp": 100, "atk": 120, "def": 100, "spa": 100, "spd": 100, "spe": 100}
        stats = {"atk": 236, "def": 236, "spa": 236, "spd": 236, "spe": 236}
        stats["spe"] = rng.choice([180, 220, 260, 300, 340])
        return NS(name=name, species=name, base_species=name, types=types, current_hp=max(0, int(300 * hp)), max_hp=300,
                   current_hp_fraction=hp, status=rng.choice(["", "", "", "tox", "brn"]), item="Leftovers", ability="",
                   level=100, base_stats=base, stats=stats, boosts={"atk": 0, "spa": 0, "spe": 0}, moves=moves,
                   fainted=hp <= 0, active=active)

    cases = []
    for i in range(samples):
        a_name, a_types, a_moves = rng.choice(species)
        o_name, o_types, o_moves = rng.choice(species)
        while o_name == a_name:
            o_name, o_types, o_moves = rng.choice(species)
        own = mon(a_name, a_types, a_moves, rng.uniform(0.12, 1.0), True)
        opp = mon(o_name, o_types, o_moves, rng.uniform(0.15, 1.0), True)
        teammates = [mon(*rng.choice(species), rng.uniform(0.25, 1.0)) for _ in range(2)]
        teammates[0].active = False
        teammates[1].active = False
        battle = NS(
            battle_tag=f"synthetic-{seed}-{i}", turn=rng.randint(2, 80), format="gen3ou",
            player_username="synthetic", opponent_username="synthetic", active_pokemon=own,
            opponent_active_pokemon=opp, available_moves=list(own.moves.values()), available_switches=teammates,
            team={p.name + str(j): p for j, p in enumerate([own] + teammates)},
            opponent_team={"opp": opp}, force_switch=False, reviving=False, can_tera=None, weather={},
            side_conditions={}, opponent_side_conditions={}, fields={}
        )
        legal = _legal_actions(battle)
        if len(legal) < 2:
            continue
        base_action = rng.choice(legal)
        predictive_action = rng.choice(legal)
        if predictive_action == base_action:
            continue
        matrix = TEAM_MATRIX_NAMES[i % len(TEAM_MATRIX_NAMES)]
        cases.append((matrix, battle, base_action, predictive_action))
    return cases


def run_synthetic(samples: int, seed: int = 7) -> dict[str, Any]:
    results = []
    for matrix, battle, base, predictive in generate_synthetic_cases(samples, seed):
        results.append(evaluate_actions(battle, base, predictive, matrix=matrix))
    negatives = [r for r in results if r.expected_delta < -2.0]
    false_predictions = [r for r in results if r.false_prediction]
    sacrifices = [r for r in results if r.win_condition_sacrifice]
    self_kos = [r for r in results if r.self_ko_override]
    by_matrix: dict[str, dict[str, float]] = {}
    for matrix in TEAM_MATRIX_NAMES:
        rows = [r for r in results if r.matrix == matrix]
        if not rows:
            continue
        by_matrix[matrix] = {
            "cases": len(rows),
            "negative_delta_rate": sum(r.expected_delta < -2.0 for r in rows) / len(rows),
            "mean_delta": sum(r.expected_delta for r in rows) / len(rows),
            "false_prediction_rate": sum(r.false_prediction for r in rows) / len(rows),
            "sacrifice_rate": sum(r.win_condition_sacrifice for r in rows) / len(rows),
        }
    return {
        "mode": "synthetic",
        "seed": seed,
        "requested_samples": samples,
        "evaluated_cases": len(results),
        "negative_delta_cases": len(negatives),
        "false_prediction_cases": len(false_predictions),
        "win_condition_sacrifice_cases": len(sacrifices),
        "self_ko_overrides": len(self_kos),
        "mean_delta": sum(r.expected_delta for r in results) / len(results) if results else 0.0,
        "worst_delta": min((r.expected_delta for r in results), default=0.0),
        "by_matrix": by_matrix,
        "worst_cases": [asdict(r) for r in sorted(results, key=lambda r: (r.expected_delta, r.worst_likely_delta))[:50]],
    }


def run_recorded_report(database: Path, limit: int = 0) -> dict[str, Any]:
    rows = audit_recorded(database, limit)
    negatives = [r for r in rows if r.expected_delta < -2.0]
    false_predictions = [r for r in rows if r.false_prediction]
    sacrifices = [r for r in rows if r.win_condition_sacrifice]
    self_kos = [r for r in rows if r.self_ko_override]
    return {
        "mode": "recorded",
        "database": str(database),
        "evaluated_overrides": len(rows),
        "negative_delta_cases": len(negatives),
        "false_prediction_cases": len(false_predictions),
        "win_condition_sacrifice_cases": len(sacrifices),
        "self_ko_overrides": len(self_kos),
        "mean_delta": sum(r.expected_delta for r in rows) / len(rows) if rows else 0.0,
        "worst_delta": min((r.expected_delta for r in rows), default=0.0),
        "worst_cases": [asdict(r) for r in sorted(rows, key=lambda r: (r.expected_delta, r.worst_likely_delta))[:100]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast offline Gen 3 counterfactual audit")
    parser.add_argument("--database", default="battle_data/battles.db")
    parser.add_argument("--recorded-limit", type=int, default=0)
    parser.add_argument("--synthetic", type=int, default=0, help="generate this many synthetic cases")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="battle_data/counterfactual_gen3_report.json")
    args = parser.parse_args()

    payload: dict[str, Any] = {"recorded": run_recorded_report(Path(args.database), args.recorded_limit)}
    if args.synthetic > 0:
        payload["synthetic"] = run_synthetic(args.synthetic, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "recorded": {k: payload["recorded"][k] for k in ("evaluated_overrides", "negative_delta_cases", "false_prediction_cases", "win_condition_sacrifice_cases", "self_ko_overrides", "mean_delta", "worst_delta")},
        "synthetic": {k: payload["synthetic"][k] for k in ("evaluated_cases", "negative_delta_cases", "false_prediction_cases", "win_condition_sacrifice_cases", "self_ko_overrides", "mean_delta", "worst_delta")} if "synthetic" in payload else None,
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
