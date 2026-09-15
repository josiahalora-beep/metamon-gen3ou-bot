import unittest
from types import SimpleNamespace

from battle_ai.damage import calculate_damage


class Move:
    def __init__(self, name, power, typ):
        self.id = name
        self.name = name
        self.base_power = power
        self.type = SimpleNamespace(name=typ)


def mon(name, types, stats, base_stats, current_hp, max_hp):
    return SimpleNamespace(
        name=name,
        species=name,
        base_species=name,
        types=types,
        stats=stats,
        base_stats=base_stats,
        level=100,
        current_hp=current_hp,
        max_hp=max_hp,
        current_hp_fraction=current_hp / max_hp,
        status="",
        boosts={},
    )


class DamagePercentageTests(unittest.TestCase):
    def test_damage_percentages_use_current_hp(self):
        attacker = mon(
            "claydol",
            ("ground", "psychic"),
            {"atk": 120, "def": 200, "spa": 120, "spd": 200, "spe": 100},
            {"atk": 70, "def": 105, "spa": 70, "spd": 120, "spe": 75},
            323,
            323,
        )
        defender = mon(
            "tyranitar",
            ("rock", "dark"),
            {"atk": 220, "def": 200, "spa": 120, "spd": 150, "spe": 100},
            {"atk": 134, "def": 110, "spa": 95, "spd": 100, "spe": 61},
            150,
            300,
        )
        result = calculate_damage(attacker, defender, Move("explosion", 250, "Normal"))
        self.assertGreater(result.max_damage, 0)
        self.assertAlmostEqual(result.percentage_min, 100.0 * result.min_damage / 150.0)
        self.assertAlmostEqual(result.percentage_max, 100.0 * result.max_damage / 150.0)


if __name__ == "__main__":
    unittest.main()
