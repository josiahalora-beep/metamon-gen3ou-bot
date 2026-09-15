"""Offline counterfactual benchmark for SyntheticRLV2 vs its tactical verifier.

This never connects to Showdown. Recorded SQLite turns provide the actual
SyntheticRLV2 action (A); the current verifier is evaluated on the reconstructed
same state (B). Deterministic fixture positions test hard facts separately.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS

from battle_ai.damage import calculate_damage
from battle_ai.evaluator import TacticalEvaluator
from metamon.interface import consistent_move_order, consistent_pokemon_order


def _move(data):
    return NS(id=data.get("id", ""), name=data.get("name", data.get("id", "")),
              base_power=int(data.get("base_power", 0) or 0),
              type=NS(name=data.get("type", "")), category=NS(name=data.get("category", "")),
              priority=int(data.get("priority", 0) or 0))


def _pokemon(data):
    moves = {_move(m).id: _move(m) for m in data.get("moves", [])}
    return NS(name=data.get("name", ""), species=data.get("name", ""),
              base_species=data.get("base_species", data.get("name", "")),
              types=tuple(data.get("types", [])), current_hp=data.get("hp", 0),
              max_hp=data.get("max_hp", 0), current_hp_fraction=data.get("hp_fraction", 0),
              status=data.get("status", ""), item=data.get("item", ""), ability=data.get("ability", ""),
              level=data.get("level", 100), base_stats=data.get("base_stats", {}),
              stats=data.get("stats", {}), boosts=data.get("boosts", {}), moves=moves,
              fainted=data.get("fainted", False), active=False)


def battle_from_snapshot(snapshot: dict):
    active = _pokemon(snapshot.get("our_active", {})); active.active = True
    target = _pokemon(snapshot.get("opponent_active", {})); target.active = True
    our_team = [_pokemon(p) for p in snapshot.get("our_team", [])]
    if not our_team or all(p.name != active.name for p in our_team):
        our_team.insert(0, active)
    opponent_team = [_pokemon(p) for p in snapshot.get("opponent_team", [])]
    legal_moves = list(active.moves.values())
    switches = [p for p in our_team if p.name != active.name and not p.fainted]
    return NS(
        battle_tag=snapshot.get("battle_id", "offline"), turn=snapshot.get("turn", 0),
        format=snapshot.get("format", "gen3ou"), player_username=snapshot.get("player", ""),
        opponent_username=snapshot.get("opponent", ""), active_pokemon=active,
        opponent_active_pokemon=target, available_moves=legal_moves,
        available_switches=switches, team={p.name: p for p in our_team},
        opponent_team={p.name: p for p in opponent_team}, force_switch=snapshot.get("forced_switch", False),
        reviving=False, can_tera=None, weather={str(w): 1 for w in snapshot.get("weather", [])},
        side_conditions={}, opponent_side_conditions={}, fields={},
    )


def hard_fact_for_action(battle, action: int) -> dict:
    """Return only mechanical facts; heuristic scores are never called facts."""
    if action < 0 or action >= 9:
        return {"legal": False}
    if action >= 4:
        try:
            switches = consistent_pokemon_order(list(battle.available_switches))
        except ValueError:
            switches = sorted(battle.available_switches, key=lambda p: str(getattr(p, "name", "")))
        legal = 0 <= action - 4 < len(switches)
        return {"legal": legal, "kind": "switch", "hard": []}
    try:
        moves = consistent_move_order(list(battle.active_pokemon.moves.values()))
    except ValueError:
        moves = sorted(battle.active_pokemon.moves.values(), key=lambda m: str(getattr(m, "id", "")))
    if action >= len(moves):
        return {"legal": False, "kind": "move", "hard": []}
    move = moves[action]
    if move.id not in {m.id for m in battle.available_moves}:
        return {"legal": False, "kind": "move", "hard": []}
    weather = next(iter(battle.weather), "")
    weather = str(getattr(weather, "name", weather))
    result = calculate_damage(battle.active_pokemon, battle.opponent_active_pokemon, move, weather=weather)
    facts = []
    if result.reliable and result.ko_probability == 1.0:
        facts.append("guaranteed_ko")
    if result.reliable and result.min_damage * 2 >= getattr(battle.opponent_active_pokemon, "current_hp", 0):
        facts.append("guaranteed_2hko")
    return {"legal": True, "kind": "move", "hard": facts,
            "damage": result.__dict__, "incoming_ko_risk": None}


def _override_classification(model_action: int, final_action: int, model_legal: bool,
                             final_hard_facts: list[str] | None, evaluations: list[dict]) -> str:
    if model_action == final_action:
        return "none"
    if not model_legal:
        return "illegal_model_action"
    if final_hard_facts:
        return "hard_fact"
    final_eval = next((e for e in evaluations if int(e.get("action", -1)) == final_action), None)
    reason = str(final_eval.get("reason", "")) if final_eval else ""
    if "anti-throw:" in reason or "setup emergency:" in reason:
        return "strategic_safety"
    return "unjustified_heuristic"


def run_recorded(database: Path, limit: int) -> list[dict]:
    db = sqlite3.connect(database)
    try:
        rows = db.execute("SELECT battle_id,turn,snapshot_json,candidate_actions_json,chosen_action FROM turns ORDER BY id LIMIT ?", (limit,)).fetchall()
        evaluator = TacticalEvaluator()
        results = []
        for battle_id, turn, raw, raw_candidates, model_action in rows:
            snapshot = json.loads(raw)
            battle = battle_from_snapshot(snapshot)
            try:
                legal = sorted(int(entry["action"]) for entry in json.loads(raw_candidates or "[]"))
            except (TypeError, ValueError, json.JSONDecodeError, KeyError):
                legal = list(range(4)) + [4 + i for i in range(max(0, len(battle.available_switches)))]
            final, evaluations = evaluator.evaluate(battle, legal, int(model_action))
            eval_rows = [e.__dict__ for e in evaluations]
            model_fact = hard_fact_for_action(battle, int(model_action))
            final_fact = hard_fact_for_action(battle, int(final))
            model_legal = int(model_action) in legal
            classification = _override_classification(
                int(model_action), int(final), model_legal,
                final_fact.get("hard", []), eval_rows,
            )
            results.append({
                "source": "recorded", "battle_id": battle_id, "turn": turn,
                "model_action": int(model_action), "final_action": int(final),
                "override": int(model_action) != int(final),
                "override_classification": classification,
                "model_legal": model_legal, "final_legal": int(final) in legal,
                "legal_actions": legal,
                "model_score": None,
                "position_evaluation": None,
                "model_damage": model_fact.get("damage"),
                "final_damage": final_fact.get("damage"),
                "incoming_ko_risk": final_fact.get("incoming_ko_risk"),
                "model_hard_facts": model_fact.get("hard", []), "final_hard_facts": final_fact.get("hard", []),
                "evaluations": eval_rows,
                "tactical_confidence": "high" if final_fact.get("hard") else "unknown",
            })
        return results
    finally:
        db.close()


@dataclass(frozen=True)
class Fixture:
    name: str
    expected_hard_fact: str
    model_action: int
    battle: object


def _fixture(name, expected, model_action, attacker_stats, defender_stats, move_power=100, move_type="Normal", defender_types=("Normal",)):
    move = NS(id="fixturemove", name="fixturemove", base_power=move_power, type=NS(name=move_type), category=NS(name="Physical"), priority=0)
    active = NS(name="attacker", species="attacker", base_species="attacker", types=(move_type.lower(),), current_hp=300, max_hp=300, current_hp_fraction=1.0, status="", item="", ability="", level=100, base_stats={}, stats=attacker_stats, boosts={}, moves={move.id: move}, fainted=False, active=True)
    target = NS(name="defender", species="defender", base_species="defender", types=defender_types, current_hp=300, max_hp=300, current_hp_fraction=1.0, status="", item="", ability="", level=100, base_stats={}, stats=defender_stats, boosts={}, moves={}, fainted=False, active=True)
    return Fixture(name, expected, model_action, NS(battle_tag=name, turn=1, format="gen3ou", player_username="a", opponent_username="b", active_pokemon=active, opponent_active_pokemon=target, available_moves=[move], available_switches=[], team={"attacker": active}, opponent_team={"defender": target}, force_switch=False, reviving=False, can_tera=None, weather={}, side_conditions={}, opponent_side_conditions={}, fields={}))


def deterministic_fixtures() -> list[Fixture]:
    normal = {"atk": 200, "def": 200, "spa": 200, "spd": 200, "spe": 200}
    huge = {"atk": 500, "def": 50, "spa": 500, "spd": 50, "spe": 400}
    return [
        _fixture("guaranteed_ohko", "guaranteed_ko", 0, huge, {"atk": 50, "def": 50, "spa": 50, "spd": 50, "spe": 50}),
        _fixture("guaranteed_2hko", "guaranteed_2hko", 0, normal, {"atk": 50, "def": 100, "spa": 50, "spd": 100, "spe": 50}),
        _fixture("opponent_guaranteed_ohko", "incoming_ko_unavailable_without_opponent_move", 0, normal, huge),
        _fixture("faster_attacker", "speed_order", 0, {**normal, "spe": 400}, normal),
        _fixture("safe_switch", "safe_switch", 4, normal, normal),
        _fixture("dangerous_switch", "dangerous_switch", 4, normal, huge),
        _fixture("setup_sweeper", "setup_threat", 0, normal, normal),
        _fixture("final_counter_preservation", "preserve_counter", 4, normal, huge),
        _fixture("explosion_winning_trade", "winning_trade", 0, huge, normal, move_power=250),
        _fixture("unnecessary_explosion", "avoid_explosion", 0, normal, normal, move_power=250),
        _fixture("weather_preservation", "weather", 0, normal, normal, move_type="Water"),
        _fixture("hazard_punishment", "hazard_cost", 0, normal, normal),
        _fixture("status_preservation", "status_value", 0, normal, normal),
        _fixture("endgame_win_condition", "endgame", 0, huge, normal),
    ]


def run_fixtures() -> list[dict]:
    evaluator = TacticalEvaluator()
    out = []
    for fixture in deterministic_fixtures():
        legal = [0, 4]
        final, evaluations = evaluator.evaluate(fixture.battle, legal, fixture.model_action)
        facts = hard_fact_for_action(fixture.battle, final)
        out.append({"source": "fixture", "name": fixture.name, "expected_hard_fact": fixture.expected_hard_fact,
                    "model_action": fixture.model_action, "final_action": final, "override": final != fixture.model_action,
                    "model_score": None, "position_evaluation": None,
                    "model_damage": facts.get("damage"), "final_damage": facts.get("damage"),
                    "incoming_ko_risk": facts.get("incoming_ko_risk"),
                    "hard_facts": facts.get("hard", []), "evaluations": [e.__dict__ for e in evaluations]})
    return out


def report(recorded, fixtures):
    overrides = [r for r in recorded if r["override"]]
    justified = [r for r in overrides if r.get("final_hard_facts") or not r.get("model_legal", True)]
    reasons = {}
    for row in overrides:
        reason = row.get("override_classification", "unclassified")
        reasons[reason] = reasons.get(reason, 0) + 1
    strategic = sum(r.get("override_classification") == "strategic_safety" for r in recorded)
    hard = sum(r.get("override_classification") == "hard_fact" for r in recorded)
    unjustified = sum(r.get("override_classification") == "unjustified_heuristic" for r in recorded)
    illegal = sum(r.get("override_classification") == "illegal_model_action" for r in recorded)
    return {
        "recorded_states": len(recorded), "fixture_states": len(fixtures),
        "preserved_percentage": (100 * (len(recorded) - len(overrides)) / len(recorded)) if recorded else None,
        "overridden_percentage": (100 * len(overrides) / len(recorded)) if recorded else None,
        "override_count": len(overrides),
        "hard_fact_justified_overrides": len(justified),
        "hard_fact_justification_percentage": (100 * len(justified) / len(overrides)) if overrides else None,
        "strategic_safety_overrides": strategic,
        "hard_fact_overrides": hard,
        "unjustified_heuristic_overrides": unjustified,
        "illegal_model_overrides": illegal,
        "override_reasons": reasons,
        "model_illegal_rate": 100 * sum(not r["model_legal"] for r in recorded) / len(recorded) if recorded else None,
        "verifier_illegal_rate": 100 * sum(not r["final_legal"] for r in recorded) / len(recorded) if recorded else None,
        "obvious_blunder_rate_hard_facts_only": 100 * sum(not r["model_legal"] for r in recorded) / len(recorded) if recorded else None,
        "guaranteed_ko_recognition": sum("guaranteed_ko" in r.get("hard_facts", []) or "guaranteed_ko" in r.get("final_hard_facts", []) for r in fixtures),
        "guaranteed_2hko_recognition": sum("guaranteed_2hko" in r.get("hard_facts", []) or "guaranteed_2hko" in r.get("final_hard_facts", []) for r in fixtures),
        "guaranteed_loss_avoidance": "not implemented: incoming KO evaluator is intentionally unavailable",
        "rows": recorded + fixtures,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--database", default="battle_data/battles.db")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--output", default="battle_data/counterfactual_report.json")
    ap.add_argument("--text-output", default="battle_data/counterfactual_report.txt")
    args = ap.parse_args()
    recorded = run_recorded(Path(args.database), args.limit)
    fixtures = run_fixtures()
    result = report(recorded, fixtures)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    lines = ["SyntheticRLV2 counterfactual benchmark", "", f"Recorded states: {result['recorded_states']}", f"Fixture states: {result['fixture_states']}", f"Actions preserved: {result['preserved_percentage']}", f"Actions overridden: {result['overridden_percentage']}", f"Hard-fact justified overrides: {result['hard_fact_justification_percentage']}", f"Strategic safety overrides: {result['strategic_safety_overrides']}", f"Hard-fact overrides: {result['hard_fact_overrides']}", f"Unjustified heuristic overrides: {result['unjustified_heuristic_overrides']}", f"Override reasons: {result['override_reasons']}", f"Model illegal rate: {result['model_illegal_rate']}", f"Verifier illegal rate: {result['verifier_illegal_rate']}", f"Hard-fact-only obvious-blunder rate: {result['obvious_blunder_rate_hard_facts_only']}", f"Guaranteed-KO fixture recognition: {result['guaranteed_ko_recognition']}", f"Guaranteed-2HKO fixture recognition: {result['guaranteed_2hko_recognition']}", f"Guaranteed-loss avoidance: {result['guaranteed_loss_avoidance']}"]
    Path(args.text_output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__": main()
