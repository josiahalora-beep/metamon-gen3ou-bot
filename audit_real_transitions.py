from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

from audit_override_sources import classify_reason
from battle_ai.evaluator import TacticalEvaluator
from benchmark_counterfactual import battle_from_snapshot
from counterfactual_gen3 import _legal_actions


def _actual_transition(before: dict, after: dict) -> dict:
    our_b = before.get("our_active", {}) or {}
    our_a = after.get("our_active", {}) or {}
    opp_b = before.get("opponent_active", {}) or {}
    opp_a = after.get("opponent_active", {}) or {}
    ob_name = str(our_b.get("name", "")).lower()
    oa_name = str(our_a.get("name", "")).lower()
    pb_name = str(opp_b.get("name", "")).lower()
    pa_name = str(opp_a.get("name", "")).lower()
    our_switched = bool(ob_name and oa_name and ob_name != oa_name)
    opp_switched = bool(pb_name and pa_name and pb_name != pa_name)
    ob_hp = float(our_b.get("hp_fraction", 0.0) or 0.0)
    oa_hp = float(our_a.get("hp_fraction", 0.0) or 0.0)
    pb_hp = float(opp_b.get("hp_fraction", 0.0) or 0.0)
    pa_hp = float(opp_a.get("hp_fraction", 0.0) or 0.0)
    our_hp_loss = max(0.0, ob_hp - oa_hp) if not our_switched else 0.0
    opp_hp_loss = max(0.0, pb_hp - pa_hp) if not opp_switched else 0.0
    ob_status = str(our_b.get("status", "") or "").lower()
    oa_status = str(our_a.get("status", "") or "").lower()
    pb_status = str(opp_b.get("status", "") or "").lower()
    pa_status = str(opp_a.get("status", "") or "").lower()
    our_status_changed = ob_status != oa_status
    opp_status_changed = pb_status != pa_status
    our_fainted = bool(our_a.get("fainted", False))
    opp_fainted = bool(opp_a.get("fainted", False))
    if opp_switched:
        response = "switch"
    elif our_hp_loss > 0 or our_fainted or our_status_changed:
        response = "observed_pressure_or_effect"
    elif opp_hp_loss > 0 or opp_fainted or opp_status_changed:
        response = "observed_effect_on_opponent"
    else:
        response = "no_observable_change"
    return {
        "our_switched": our_switched,
        "opponent_switched": opp_switched,
        "opponent_response": response,
        "our_hp_before": ob_hp,
        "our_hp_after": oa_hp,
        "our_hp_loss": our_hp_loss,
        "opponent_hp_before": pb_hp,
        "opponent_hp_after": pa_hp,
        "opponent_hp_loss": opp_hp_loss,
        "our_fainted": our_fainted,
        "opponent_fainted": opp_fainted,
        "our_status_before": ob_status,
        "our_status_after": oa_status,
        "opponent_status_before": pb_status,
        "opponent_status_after": pa_status,
        "our_active_before": ob_name,
        "our_active_after": oa_name,
        "opponent_active_before": pb_name,
        "opponent_active_after": pa_name,
    }


def _switch_target_name(battle, action: int) -> str:
    if action < 4:
        return ""
    slots = list(getattr(battle, "available_switches", []) or [])
    idx = action - 4
    if 0 <= idx < len(slots):
        return str(getattr(slots[idx], "species", getattr(slots[idx], "name", "")))
    return ""


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
                legal = [int(x["action"]) for x in json.loads(raw_candidates or "[]")]
            except (TypeError, ValueError, json.JSONDecodeError, KeyError):
                legal = list(_legal_actions(battle))
            decoded.append({"id": int(row_id), "battle_id": str(battle_id), "turn": int(turn),
                            "snapshot": snapshot, "battle": battle, "legal": legal,
                            "model_action": int(model_action)})
        except Exception:
            continue

    evaluator = TacticalEvaluator()
    records = []
    errors = 0
    for i, current in enumerate(decoded[:-1]):
        nxt = decoded[i + 1]
        if nxt["battle_id"] != current["battle_id"] or nxt["turn"] != current["turn"] + 1:
            continue
        try:
            legal = current["legal"] or list(_legal_actions(current["battle"]))
            model_action = current["model_action"]
            if model_action not in legal:
                continue
            final_action, evaluations = evaluator.evaluate(current["battle"], legal, model_action)
            if int(final_action) == model_action:
                continue
            chosen_eval = next((e for e in evaluations if int(e.action) == int(final_action)), None)
            reason = chosen_eval.reason if chosen_eval else ""
            transition = _actual_transition(current["snapshot"], nxt["snapshot"])
            target = _switch_target_name(current["battle"], int(final_action))
            switch_match = None
            if final_action >= 4:
                switch_match = (
                    transition["our_switched"]
                    and transition["our_active_after"].replace("-", "") == target.lower().replace("-", "")
                )
            records.append({"battle_id": current["battle_id"], "turn": current["turn"],
                            "model_action": model_action, "final_action": int(final_action),
                            "source": classify_reason(reason), "reason": reason,
                            "final_kind": "switch" if final_action >= 4 else "move",
                            "switch_target": target, "transition": transition,
                            "switch_execution_match": switch_match})
        except Exception:
            errors += 1

    source_counts = Counter(r["source"] for r in records)
    response_counts = Counter(r["transition"]["opponent_response"] for r in records)
    switches = [r for r in records if r["final_kind"] == "switch" and r["switch_execution_match"] is not None]
    matched = [r for r in switches if r["switch_execution_match"]]
    payload = {
        "evaluated_real_transitions": len(records),
        "errors": errors,
        "source_counts": dict(source_counts),
        "actual_opponent_responses": dict(response_counts),
        "switch_overrides": len(switches),
        "switch_execution_matches": len(matched),
        "switch_execution_failures": len(switches) - len(matched),
        "switch_execution_match_rate": (len(matched) / len(switches)) if switches else None,
        "switches_followed_by_observed_pressure": sum(r["transition"]["opponent_response"] == "observed_pressure_or_effect" for r in matched),
        "successful_switch_survival": sum(not r["transition"]["our_fainted"] for r in matched),
        "move_overrides": sum(r["final_kind"] == "move" for r in records),
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
    summary = {k: payload[k] for k in (
        "evaluated_real_transitions", "errors", "source_counts", "actual_opponent_responses",
        "switch_overrides", "switch_execution_matches", "switch_execution_failures",
        "switch_execution_match_rate", "switches_followed_by_observed_pressure",
        "successful_switch_survival", "move_overrides"
    )}
    summary["output"] = args.output
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
