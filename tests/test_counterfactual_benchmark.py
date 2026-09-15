import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from battle_ai.state import snapshot_battle
from benchmark_counterfactual import (
    battle_from_snapshot,
    deterministic_fixtures,
    hard_fact_for_action,
    report,
    run_fixtures,
    run_recorded,
)


class CounterfactualBenchmarkTests(unittest.TestCase):
    def test_fixture_catalog_covers_required_cases(self):
        names = {fixture.name for fixture in deterministic_fixtures()}
        self.assertGreaterEqual(len(names), 14)
        for required in {
            "guaranteed_ohko", "guaranteed_2hko", "opponent_guaranteed_ohko",
            "safe_switch", "dangerous_switch", "setup_sweeper",
            "final_counter_preservation", "explosion_winning_trade",
            "unnecessary_explosion", "weather_preservation", "hazard_punishment",
            "status_preservation", "endgame_win_condition",
        }:
            self.assertIn(required, names)

    def test_hard_facts_are_mechanical_and_soft_cases_are_not_claimed(self):
        rows = {row["name"]: row for row in run_fixtures()}
        self.assertIn("guaranteed_ko", rows["guaranteed_ohko"]["hard_facts"])
        self.assertIn("guaranteed_2hko", rows["guaranteed_2hko"]["hard_facts"])
        self.assertNotIn("incoming_guaranteed_ko", rows["opponent_guaranteed_ohko"]["hard_facts"])
        self.assertEqual(rows["opponent_guaranteed_ohko"]["expected_hard_fact"],
                         "incoming_ko_unavailable_without_opponent_move")

    def test_recorded_benchmark_uses_exact_recorded_candidate_actions(self):
        fixture = deterministic_fixtures()[0]
        snapshot = snapshot_battle(fixture.battle, [0, 4]).to_dict()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "turns.db"
            db = sqlite3.connect(path)
            db.execute("CREATE TABLE turns (id INTEGER PRIMARY KEY, battle_id TEXT, turn INTEGER, snapshot_json TEXT, candidate_actions_json TEXT, chosen_action INTEGER)")
            db.execute("INSERT INTO turns VALUES (1, ?, ?, ?, ?, ?)",
                       ("fixture", 1, json.dumps(snapshot), json.dumps([{"action": 4}]), 0))
            db.commit()
            db.close()
            rows = run_recorded(path, 10)
        self.assertEqual(rows[0]["legal_actions"], [4])
        self.assertFalse(rows[0]["model_legal"])
        self.assertTrue(rows[0]["final_legal"])
        self.assertEqual(rows[0]["final_action"], 4)

    def test_report_does_not_call_soft_estimates_improvement(self):
        result = report([], run_fixtures())
        self.assertEqual(result["recorded_states"], 0)
        self.assertEqual(result["override_reasons"], {})
        self.assertEqual(result["guaranteed_loss_avoidance"],
                         "not implemented: incoming KO evaluator is intentionally unavailable")

    def test_invalid_switch_is_not_a_hard_fact_or_legal_action(self):
        fixture = deterministic_fixtures()[0]
        self.assertFalse(hard_fact_for_action(fixture.battle, 8)["legal"])


if __name__ == "__main__":
    unittest.main()
