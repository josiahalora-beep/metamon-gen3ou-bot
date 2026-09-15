import unittest

from benchmark_counterfactual import _override_classification


class CounterfactualClassificationTests(unittest.TestCase):
    def test_strategic_safety_is_distinct_from_hard_fact(self):
        evaluations = [
            {"action": 5, "reason": "switch position not numerically evaluated | anti-throw: active celebi is compromised"}
        ]
        self.assertEqual(
            _override_classification(2, 5, True, [], evaluations),
            "strategic_safety",
        )

    def test_hard_fact_remains_distinct(self):
        evaluations = [{"action": 0, "reason": "guaranteed KO"}]
        self.assertEqual(
            _override_classification(1, 0, True, ["guaranteed_ko"], evaluations),
            "hard_fact",
        )

    def test_unjustified_heuristic_is_explicit(self):
        evaluations = [{"action": 5, "reason": "some heuristic"}]
        self.assertEqual(
            _override_classification(2, 5, True, [], evaluations),
            "unjustified_heuristic",
        )


if __name__ == "__main__":
    unittest.main()
