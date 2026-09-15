import unittest
from types import SimpleNamespace

from battle_ai.damage import calculate_damage
from battle_ai.estimation import stat_range


class Move:
    def __init__(self, move_id, power, move_type):
        self.id = move_id
        self.name = move_id
        self.base_power = power
        self.type = SimpleNamespace(name=move_type)


def mon(name, types, base_stats, stats, hp_fraction=1.0):
    return SimpleNamespace(
        name=name, species=name, types=types, base_stats=base_stats, stats=stats,
        level=100, current_hp_fraction=hp_fraction, max_hp=100,
        current_hp=max(1, int(100 * hp_fraction)), boosts={}, status="", moves={},
    )


class DamageEstimationTests(unittest.TestCase):
    def test_hidden_stat_range_uses_species_base_stats(self):
        blissey = mon("blissey", ("normal",), {"hp": 255, "spa": 75, "spd": 135},
                      {"spa": None, "spd": None}, 0.57)
        lo, hi = stat_range(blissey, "spd")
        self.assertLess(lo, hi)
        self.assertGreater(hi, 200)

    def test_unknown_stats_get_conservative_estimate(self):
        attacker = mon("swampert", ("water", "ground"),
                        {"hp": 100, "atk": 110, "def": 90, "spa": 85, "spd": 90, "spe": 60},
                        {"atk": 344})
        defender = mon("skarmory", ("steel", "flying"),
                        {"hp": 65, "atk": 80, "def": 140, "spa": 40, "spd": 70, "spe": 70},
                        {"def": None, "spd": None}, 0.13)
        result = calculate_damage(attacker, defender, Move("surf", 95, "Water"))
        self.assertTrue(result.reliable)
        self.assertGreater(result.max_damage, 0)
        self.assertIn("estimated", result.reason)

    def test_unknown_fixture_without_species_stats_stays_unknown(self):
        attacker = mon("unknown", ("normal",), {}, {"atk": 200})
        defender = mon("unknown", ("normal",), {}, {"def": None}, 1.0)
        result = calculate_damage(attacker, defender, Move("tackle", 35, "Normal"))
        self.assertFalse(result.reliable)


if __name__ == "__main__":
    unittest.main()
