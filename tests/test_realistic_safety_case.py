import unittest
from types import SimpleNamespace

from battle_ai.strategic import safety_override


class Move:
    def __init__(self, name, power, typ, category="Special"):
        self.id = name
        self.name = name
        self.base_power = power
        self.type = SimpleNamespace(name=typ)
        self.category = SimpleNamespace(name=category)
        self.priority = 0


def mon(name, types, hp, max_hp, moves=(), boosts=None, stats=None, status=""):
    return SimpleNamespace(
        name=name,
        species=name,
        base_species=name,
        types=types,
        current_hp=hp,
        max_hp=max_hp,
        current_hp_fraction=hp / max_hp,
        status=status,
        item="leftovers",
        ability="",
        level=100,
        base_stats={},
        stats=stats or {"atk": 200, "def": 250, "spa": 300, "spd": 250, "spe": 250},
        boosts=boosts or {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
        moves={m.id: m for m in moves},
        fainted=False,
        active=False,
    )


class ObservedSafetyCaseTests(unittest.TestCase):
    def test_celebi_should_be_protected_from_revealed_boosted_blissey_ice_beam(self):
        celebi_moves = (
            Move("gigadrain", 60, "Grass"),
            Move("hiddenpowerfire", 70, "Fire"),
            Move("leechseed", 0, "Grass", "Status"),
            Move("psychic", 90, "Psychic"),
        )
        celebi = mon(
            "celebi",
            ("Psychic", "Grass"),
            77,
            389,
            moves=celebi_moves,
            stats={"atk": 186, "def": 238, "spa": 320, "spd": 236, "spe": 256},
        )
        metagross = mon(
            "metagross",
            ("Steel", "Psychic"),
            343,
            343,
            moves=(Move("earthquake", 100, "Ground", "Physical"),),
            stats={"atk": 390, "def": 300, "spa": 203, "spd": 216, "spe": 208},
        )
        skarmory = mon("skarmory", ("Steel", "Flying"), 334, 334)
        blissey = mon(
            "blissey",
            ("Normal",),
            57,
            100,
            moves=(Move("calmmind", 0, "Psychic", "Status"), Move("icebeam", 95, "Ice")),
            boosts={"atk": 0, "def": 0, "spa": 2, "spd": 0, "spe": 0},
        )
        celebi.active = True
        battle = SimpleNamespace(
            active_pokemon=celebi,
            opponent_active_pokemon=blissey,
            team={"celebi": celebi, "metagross": metagross, "skarmory": skarmory},
            force_switch=False,
            weather={"sandstorm": 1},
        )

        decision = safety_override(battle, [0, 1, 2, 3, 4, 5], 2)

        self.assertEqual(decision.action, 4)
        self.assertTrue(decision.hard)
        self.assertIn("compromised active", decision.reason)
        self.assertIn("safer", decision.reason)


if __name__ == "__main__":
    unittest.main()
