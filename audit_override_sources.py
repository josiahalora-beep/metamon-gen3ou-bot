from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

from battle_ai.opponent_model import OpponentModel
from battle_ai.evaluator import TacticalEvaluator
from benchmark_counterfactual import battle_from_snapshot
from counterfactual_gen3 import _legal_actions, evaluate_actions


def classify_reason(reason: str) -> str:
    text = (reason or "").lower()
    if "protect sequence breaker" in text:
        return "protect_sequence"
    if "response search:" in text:
        return "response_search"
    if "threat" in text or "hidden" in text:
        return "threat_response"
    if "strategic" in text or "win-condition" in text or "win condition" in text or "hazard plan:" in text:
        return "strategic"
    if "safety" in text or "hp" in text or "ko risk" in text or "dangerous" in text or "low-hp preservation" in text or "anti-throw" in text:
        return "safety"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit which TacticalEvaluator layer caused each override")
    parser.add_argument("--database", default="battle_data/battles.db")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="battle_data/override_source_audit.json")
    args = parser.parse_args()

    db = sqlite3.connect(args.database)
    try:
        sql = "SELECT battle_id,turn,snapshot_json,chosen_action FROM turns ORDER BY id"
        rows = db.execute(sql + (" LIMIT ?" if args.limit > 0 else ""), (args.limit,) if args.limit > 0 else ()).fetchall()
    finally:
        db.close()

    evaluator = TacticalEvaluator()
    model = OpponentModel()
    records = []
    errors = 0

    for battle_id, turn, raw, base_action in rows:
        try:
            battle = battle_from_snapshot(json.loads(raw))
            legal = _legal_actions(battle)
            if int(base_action) not in legal:
                continue
            final, evaluations = evaluator.evaluate(battle, legal, int(base_action))
            if int(final) == int(base_action):
                continue
            chosen_eval = next((e for e in evaluations if int(e.action) == int(final)), None)
            reason = chosen_eval.reason if chosen_eval else ""
            source = classify_reason(reason)
            cf = evaluate_actions(battle, int(base_action), int(final), matrix="recorded", model=model)
            records.append({
                "battle_id": battle_id,
                "turn": int(turn),
                "base_action": int(base_action),
                "final_action": int(final),
                "source": source,
                "reason": reason,
                "expected_delta": cf.expected_delta,
                "worst_likely_delta": cf.worst_likely_delta,
                "false_prediction_proxy": cf.false_prediction,
                "self_ko_override": cf.self_ko_override,
                "base_value": cf.base_value,
                "final_value": cf.predictive_value,
                "top_response": cf.top_response,
            })
        except Exception:
            errors += 1

    source_counts = Counter(r["source"] for r in records)
    negative_by_source = Counter(r["source"] for r in records if r["expected_delta"] < -2.0)
    mean_by_source = {}
    for source in sorted(source_counts):
        values = [r["expected_delta"] for r in records if r["source"] == source]
        mean_by_source[source] = sum(values) / len(values) if values else 0.0

    payload = {
        "evaluated_overrides": len(records),
        "errors": errors,
        "source_counts": dict(source_counts),
        "negative_delta_by_source": dict(negative_by_source),
        "mean_delta_by_source": mean_by_source,
        "worst_cases": sorted(records, key=lambda r: (r["expected_delta"], r["worst_likely_delta"]))[:100],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "evaluated_overrides": payload["evaluated_overrides"],
        "errors": payload["errors"],
        "source_counts": payload["source_counts"],
        "negative_delta_by_source": payload["negative_delta_by_source"],
        "mean_delta_by_source": payload["mean_delta_by_source"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
