import unittest
from types import SimpleNamespace as NS

from counterfactual_gen3 import TEAM_MATRIX_NAMES, CounterfactualResult, generate_synthetic_cases, run_synthetic


class CounterfactualGen3Tests(unittest.TestCase):
    def test_has_all_19_team_matrix_names(self):
        self.assertEqual(len(TEAM_MATRIX_NAMES), 19)
        self.assertEqual(len(set(TEAM_MATRIX_NAMES)), 19)

    def test_synthetic_sampler_is_seeded_and_repeatable(self):
        a = generate_synthetic_cases(60, seed=31)
        b = generate_synthetic_cases(60, seed=31)
        self.assertEqual(len(a), len(b))
        self.assertEqual(
            [(m, battle.battle_tag, base, pred) for m, battle, base, pred in a],
            [(m, battle.battle_tag, base, pred) for m, battle, base, pred in b],
        )

    def test_synthetic_report_is_deterministic(self):
        first = run_synthetic(80, seed=41)
        second = run_synthetic(80, seed=41)
        self.assertEqual(first["evaluated_cases"], second["evaluated_cases"])
        self.assertEqual(first["negative_delta_cases"], second["negative_delta_cases"])
        self.assertEqual(first["false_prediction_cases"], second["false_prediction_cases"])
        self.assertEqual(first["worst_delta"], second["worst_delta"])

    def test_result_is_serializable_and_flags_are_boolean(self):
        result = CounterfactualResult(
            matrix=TEAM_MATRIX_NAMES[0], battle_id="x", turn=4,
            base_action=4, predictive_action=0, expected_delta=-3.0,
            worst_likely_delta=-5.0, false_prediction=True,
            win_condition_sacrifice=True, self_ko_override=False,
            base_value=10.0, predictive_value=7.0, top_response="attack:eq 80%",
        )
        payload = result.__dict__
        self.assertIsInstance(payload["false_prediction"], bool)
        self.assertIsInstance(payload["win_condition_sacrifice"], bool)
        self.assertIsInstance(payload["self_ko_override"], bool)
        self.assertEqual(payload["expected_delta"], -3.0)


if __name__ == "__main__":
    unittest.main()
