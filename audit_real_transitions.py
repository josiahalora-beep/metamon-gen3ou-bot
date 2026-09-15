from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS

from battle_ai.evaluator import TacticalEvaluator
from benchmark_counterfactual import battle_from_snapshot
from counterfactual_gen3 import _legal_actions


def _hp_fraction(pokemon):
    try:
        return float(getattr(pokemon, "current_hp_fraction", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _status(pokemon):
    value = getattr(pokemon, "status", "")
    return str(getattr(value, "name", value) or value).lower()


def _species(pokemon):
    if pokemon is None:
        return ""
    return str(getattr(pokemon, "species", getattr(pokemon, "name", "")) or "").lower()


def _switch_target_name(battle, action):
    if action < 4:
        return ""
    slots = list(getattr(battle, "available_switches", []) or [])
    idx = action - 4
    return str(getattr(slots[idx], "species", getattr(slots[idx], "name", ""))) if 0 <= idx < len(slots) else ""


def _actual_transition(before, after):
    our_before = before.get("our_active", {}) or {}
    our_after = after.get("our_active", {}) or {}
    opp_before = before.get("opponent_active", {}) or {}
    opp_after = after.get("opponent_active", {}) or {}

    our_before_name = str(our_before.get("name", "")).lower()
    our_after_name = str(our_after.get("name", "")).lower()
    opp_before_name = str(opp_before.get("name", "")).lower()
    opp_after_name = str(opp_after.get("name", "")).lower()

    our_switched = bool(our_before_name and our_after_name and our_before_name != our_after_name)
    opponent_switched = bool(opp_before_name and opp_after_name and opp_before_name != opp_after_name)

    our_hp_before = float(our_before.get("hp_fraction", 0.0) or 0.0)
    our_hp_after = float(our_after.get("hp_fraction", 0.0) or 0.0)
    opp_hp_before = float(opp_before.get("hp_fraction", 0.0) or 0.0)
    opp_hp_after = float(opp_after.get("hp_fraction", 0.0) or 0.0)
    our_status_before = str(our_before.get("status", "") or "").lower()
    our_status_after = str(our_after.get("status", "") or "").lower()
    opp_status_before = str(opp_before.get("status", "") or "").lower()
    opp_status_after = str(opp_after.get("status", "") or "").lower()

    our_hp_loss = max(0.0, our_hp_before - our_hp_after) if not our_switched else 0.0
    opp_hp_loss = max(0.0, opp_hp_before - opp_hp_after) if not opponent_switched else 0.0
    our_fainted = bool(our_after.get("fainted", False))
    opp_fainted = bool(opp_after.get("fainted", False))
    our_changed_without_switch = (our_status_before != our_status_after)
    opp_changed_without_switch = (opp_status_before != opp_status_after)

    if opponent_switched:
        opponent_response = "switch"
    elif our_hp_loss > 0.0 or our_fainted or our_changed_without_switch:
        opponent_response = "non-switch_pressure"
    elif opp_hp_loss > 0.0 or opp_fainted or opp_changed_without_switch:
        opponent_response = "attack_or_effect_on_opponent"
    else:
        opponent_response = "unknown"

    return {
        "our_switched": our_switched,
        "opponent_switched": opponent_switched,
        "opponent_response": opponent_response,
        "our_hp_before": our_hp_before,
        "our_hp_after": our_hp_after,
        "our_hp_loss": our_hp_loss,
        "opponent_hp_before": opp_hp_before,
        "opponent_hp_after": opp_hp_after,
        "opponent_hp_loss": opp_hp_loss,
        "our_fainted": our_fainted,
        "opponent_fainted": opp_fainted,
        "our_status_before": our_status_before,
        "our_status_after": our_status_after,
        "opponent_status_before": opp_status_before,
        "opponent_status_after": opp_status_after,
        "our_active_before": our_before_name,
        "our_active_after": our_after_name,
        "opponent_active_before": opp_before_name,
        "opponent_active_after": opp_after_name,
    }


def audit(database: Path, output: Path, limit: int = 0) -> dict:
    db = sqlite3.connect(database)
    try:
        sql = "SELECT id,battle_id,turn,snapshot_json,candidate_actions_json,chosen_action FROM turns ORDER BY id"
        params = ()
        if limit > 0:
            sql += " LIMIT ?"
            params = (limit,)
        rows = db.execute(sql, params).fetchall()
    finally:
        db.close()

    decoded = []
    for row_id, battle_id, turn, raw, raw_candidates, model_action in rows:
        try:
            snapshot = json.loads(raw)
            battle = battle_from_snapshot(snapshot)
            try:
                candidates = [int(x["action"]) for x in json.loads(raw_candidates or "[]")]
            except (TypeError, ValueError, json.JSONDecodeError, KeyError):
                candidates = list(_legal_actions(battle))
            decoded.append({
                "id": int(row_id), "battle_id": str(battle_id), "turn": int(turn),
                "snapshot": snapshot, "battle": battle, "legal": candidates,
                "model_action": int(model_action),
            })
        except Exception:
            continue

    evaluator = TacticalEvaluator()
    records = []
    errors = 0
    for index, current in enumerate(decoded):
        next_row = None
        j = index + 1
        while j < len(decoded) and decoded[j]["battle_id"] == current["battle_id"] and decoded[j]["turn"] <= current["turn"]:
            j += 1
        if j < len(decoded) and decoded[j]["battle_id"] == current["battle_id"] and decoded[j]["turn"] == current["turn"] + 1:
            next_row = decoded[j]
        if next_row is None:
            continue

        try:
            legal = current["legal"] or list(_legal_actions(current["battle"]))
            if current["model_action"] not in legal:
                continue
            final_action, evaluations = evaluator.evaluate(current["battle"], legal, current["model_action"])
            if int(final_action) == int(current["model_action"]):
                continue

            chosen_eval = next((e for e in evaluations if int(e.action) == int(final_action)), None)
            reason = chosen_eval.reason if chosen_eval else ""
            transition = _actual_transition(current["snapshot"], next_row["snapshot"])
            target = _switch_target_name(current["battle"], int(final_action))
            if final_action >= 4:
                expected_switch_executed = transition["our_switched"] and transition["our_active_after"].replace("-", "") == target.lower().replace("-", "")
            else:
                expected_switch_executed = None

            records.append({
                "battle_id": current["battle_id"],
                "turn": current["turn"],
                "model_action": int(current["model_action"]),
                "final_action": int(final_action),
                "source": reason.split(" |")[-1].strip() if reason else "unknown",
                "reason": reason,
                "final_kind": "switch" if final_action >= 4 else "move",
                "switch_target": target,
                "transition": transition,
                "switch_execution_match": expected_switch_executed,
            })
        except Exception:
            errors += 1

    source_counts = Counter(r["final_kind"] for r in records)
    opponent_responses = Counter(r["transition"]["opponent_response"] for r in records)
    switch_overrides = [r for r in records if r["final_kind"] == "switch"]
    move_overrides = [r for r in records if r["final_kind"] == "move"]
    switch_matches = [r for r in switch_overrides if r["switch_execution_match"] is not None]

    payload = {
        "evaluated_real_transitions": len(records),
        "errors": errors,
        "override_kinds": dict(source_counts),
        "actual_opponent_responses": dict(opponent_responses),
        "switch_overrides": len(switch_overrides),
        "switch_execution_matches": sum(bool(r["switch_execution_match"]) for r in switch_matches),
        "switch_execution_match_rate": (
            sum(bool(r["switch_execution_match"]) for r in switch_matches) / len(switch_matches)
            if switch_matches else None
        ),
        "move_overrides": len(move_overrides),
        "negative_sounding_transition_cases": sum(
            1 for r in records
            if r["transition"]["our_fainted"] or r["transition"]["opponent_response"] == "non-switch_pressure"
        ),
        "rows": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit actual next-turn outcomes of TacticalEvaluator overrides")
    parser.add_argument("--database", default="battle_data/battles.db")
    parser.add_argument("--output", default="battle_data/real_transition_audit.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    payload = audit(Path(args.database), Path(args.output), args.limit)
    print(json.dumps({
        "evaluated_real_transitions": payload["evaluated_real_transitions"],
        "errors": payload["errors"],
        "override_kinds": payload["override_kinds"],
        "actual_opponent_responses": payload["actual_opponent_responses"],
        "switch_overrides": payload["switch_overrides"],
        "switch_execution_match_rate": payload["switch_execution_match_rate"],
        "move_overrides": payload["move_overrides"],
        "negative_sounding_transition_cases": payload["negative_sounding_transition_cases"],
        "output": args.output,
    }, indent=2))


if __name__ == "__main__":
    main()
