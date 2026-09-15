from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

from benchmark_counterfactual import battle_from_snapshot
from counterfactual_gen3 import _legal_actions


def _canonical_name(value):
    return str(value or "").lower().replace("-", "").replace("_", "").replace(" ", "")


def _actual_transition(before, after):
    our_before = before.get("our_active", {}) or {}
    our_after = after.get("our_active", {}) or {}
    opp_before = before.get("opponent_active", {}) or {}
    opp_after = after.get("opponent_active", {}) or {}

    our_before_name = str(our_before.get("name", "")).lower()
    our_after_name = str(our_after.get("name", "")).lower()
    opp_before_name = str(opp_before.get("name", "")).lower()
    opp_after_name = str(opp_after.get("name", "")).lower()

    our_switched = bool(
        our_before_name
        and our_after_name
        and _canonical_name(our_before_name) != _canonical_name(our_after_name)
    )
    opponent_switched = bool(
        opp_before_name
        and opp_after_name
        and _canonical_name(opp_before_name) != _canonical_name(opp_after_name)
    )

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
    our_status_changed = our_status_before != our_status_after
    opp_status_changed = opp_status_before != opp_status_after

    if opponent_switched:
        opponent_response = "switch"
    elif our_hp_loss > 0.0 or our_fainted or our_status_changed:
        opponent_response = "non-switch_pressure"
    elif opp_hp_loss > 0.0 or opp_fainted or opp_status_changed:
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


def _switch_target_name(battle, action):
    if action < 4:
        return ""
    slots = list(getattr(battle, "available_switches", []) or [])
    idx = action - 4
    if 0 <= idx < len(slots):
        return str(getattr(slots[idx], "species", getattr(slots[idx], "name", "")))
    return ""


def _find_recorded_reason(raw_candidates, final_action):
    try:
        candidates = json.loads(raw_candidates or "[]")
    except (TypeError, ValueError, json.JSONDecodeError):
        return "", ""
    for item in candidates:
        try:
            if int(item.get("action")) == int(final_action):
                return str(item.get("reason", "") or ""), str(item.get("kind", "") or "")
        except (TypeError, ValueError):
            continue
    return "", ""


def _source_from_reason(reason):
    text = str(reason or "").lower()
    if "hard-loss" in text or "safety" in text:
        return "safety"
    if "hazard plan" in text or "strategic" in text or "win condition" in text:
        return "strategic"
    if "response" in text or "prediction" in text or "switch probability" in text:
        return "response_search"
    if "threat" in text:
        return "threat_response"
    return "unknown"


def _decode_rows(database: Path, limit: int):
    db = sqlite3.connect(database)
    try:
        sql = (
            "SELECT id,battle_id,turn,snapshot_json,candidate_actions_json,"
            "chosen_action,final_action,reasoning_json FROM turns ORDER BY id"
        )
        params = ()
        if limit > 0:
            sql += " LIMIT ?"
            params = (limit,)
        raw_rows = db.execute(sql, params).fetchall()
    finally:
        db.close()

    # A logical battle turn can have multiple stored decision rows. Keep the
    # latest row for each (battle_id, turn) so the transition compares the last
    # decision state for that logical turn to the next logical turn.
    latest = {}
    decode_errors = 0
    for row_id, battle_id, turn, raw, raw_candidates, chosen_action, final_action, reasoning_json in raw_rows:
        try:
            snapshot = json.loads(raw)
            battle = battle_from_snapshot(snapshot)
            try:
                candidates = [int(x["action"]) for x in json.loads(raw_candidates or "[]")]
            except (TypeError, ValueError, json.JSONDecodeError, KeyError):
                candidates = list(_legal_actions(battle))
            key = (str(battle_id), int(turn))
            latest[key] = {
                "id": int(row_id),
                "battle_id": str(battle_id),
                "turn": int(turn),
                "snapshot": snapshot,
                "battle": battle,
                "legal": candidates,
                "model_action": int(chosen_action),
                "final_action": int(final_action) if final_action is not None else int(chosen_action),
                "reasoning_json": reasoning_json or "{}",
                "candidate_actions_json": raw_candidates or "[]",
            }
        except Exception:
            decode_errors += 1

    rows = sorted(latest.values(), key=lambda r: (r["battle_id"], r["turn"], r["id"]))
    return rows, decode_errors


def audit(database: Path, output: Path, limit: int = 0) -> dict:
    decoded, decode_errors = _decode_rows(database, limit)

    records = []
    errors = decode_errors
    duplicate_turn_count = 0

    db = sqlite3.connect(database)
    try:
        duplicate_rows = db.execute(
            "SELECT battle_id, turn, COUNT(*) FROM turns GROUP BY battle_id, turn HAVING COUNT(*) > 1"
        ).fetchall()
        duplicate_turn_count = sum(int(row[2]) - 1 for row in duplicate_rows)
    finally:
        db.close()

    by_battle = {}
    for row in decoded:
        by_battle.setdefault(row["battle_id"], []).append(row)

    for battle_id, battle_rows in by_battle.items():
        battle_rows.sort(key=lambda r: r["turn"])
        for index, current in enumerate(battle_rows[:-1]):
            next_row = battle_rows[index + 1]
            if next_row["turn"] != current["turn"] + 1:
                continue

            try:
                model_action = current["model_action"]
                final_action = current["final_action"]
                if final_action == model_action:
                    continue

                reason, recorded_kind = _find_recorded_reason(
                    current["candidate_actions_json"], final_action
                )
                if not reason:
                    try:
                        reasoning = json.loads(current["reasoning_json"] or "{}")
                        reason = str(reasoning.get("reason", "") or "")
                    except (TypeError, ValueError, json.JSONDecodeError):
                        reason = ""

                transition = _actual_transition(current["snapshot"], next_row["snapshot"])
                target = _switch_target_name(current["battle"], final_action)
                switch_match = None
                if final_action >= 4:
                    switch_match = (
                        transition["our_switched"]
                        and _canonical_name(transition["our_active_after"]) == _canonical_name(target)
                    )

                records.append({
                    "battle_id": battle_id,
                    "turn": current["turn"],
                    "db_id": current["id"],
                    "model_action": model_action,
                    "final_action": final_action,
                    "source": _source_from_reason(reason),
                    "reason": reason,
                    "recorded_kind": recorded_kind,
                    "final_kind": "switch" if final_action >= 4 else "move",
                    "switch_target": target,
                    "transition": transition,
                    "switch_execution_match": switch_match,
                })
            except Exception:
                errors += 1

    override_kinds = Counter(r["final_kind"] for r in records)
    source_counts = Counter(r["source"] for r in records)
    opponent_responses = Counter(r["transition"]["opponent_response"] for r in records)
    switch_overrides = [r for r in records if r["final_kind"] == "switch"]
    move_overrides = [r for r in records if r["final_kind"] == "move"]
    switch_checks = [r for r in switch_overrides if r["switch_execution_match"] is not None]
    matched = [r for r in switch_checks if r["switch_execution_match"]]
    pressure_after_switch = [
        r for r in matched
        if r["transition"]["opponent_response"] == "non-switch_pressure"
    ]
    survived_switch = [r for r in matched if not r["transition"]["our_fainted"]]

    payload = {
        "evaluated_real_transitions": len(records),
        "errors": errors,
        "duplicate_rows_collapsed": duplicate_turn_count,
        "override_kinds": dict(override_kinds),
        "override_sources": dict(source_counts),
        "actual_opponent_responses": dict(opponent_responses),
        "switch_overrides": len(switch_overrides),
        "switch_execution_matches": len(matched),
        "switch_execution_failures": len(switch_checks) - len(matched),
        "switch_execution_match_rate": (len(matched) / len(switch_checks)) if switch_checks else None,
        "switches_followed_by_observed_pressure": len(pressure_after_switch),
        "successful_switch_survival": len(survived_switch),
        "move_overrides": len(move_overrides),
        "negative_sounding_transition_cases": sum(
            1 for r in records
            if r["transition"]["our_fainted"]
            or r["transition"]["opponent_response"] == "non-switch_pressure"
        ),
        "rows": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit actual transitions using recorded final actions")
    parser.add_argument("--database", default="battle_data/battles.db")
    parser.add_argument("--output", default="battle_data/real_transition_audit.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    payload = audit(Path(args.database), Path(args.output), args.limit)
    print(json.dumps({
        "evaluated_real_transitions": payload["evaluated_real_transitions"],
        "errors": payload["errors"],
        "duplicate_rows_collapsed": payload["duplicate_rows_collapsed"],
        "override_kinds": payload["override_kinds"],
        "override_sources": payload["override_sources"],
        "actual_opponent_responses": payload["actual_opponent_responses"],
        "switch_overrides": payload["switch_overrides"],
        "switch_execution_match_rate": payload["switch_execution_match_rate"],
        "switches_followed_by_observed_pressure": payload["switches_followed_by_observed_pressure"],
        "successful_switch_survival": payload["successful_switch_survival"],
        "move_overrides": payload["move_overrides"],
        "negative_sounding_transition_cases": payload["negative_sounding_transition_cases"],
        "output": args.output,
    }, indent=2))


if __name__ == "__main__":
    main()
