import unittest
from types import SimpleNamespace

from battle_ai.threat_response import hidden_threat_switch_override


class FakeMove:
    def __init__(self, move_id, power=0):
        self.id = move_id
        self.name = move_id
        self.base_power = power


class ThreatResponseTests(unittest.TestCase):
    @staticmethod
    def _pokemon(name, *, types=("Normal",), base_stats=None, hp=100, boosts=None, moves=None):
        return SimpleNamespace(
            name=name,
            species=name,
            types=types,
            base_stats=base_stats or {},
            stats={},
            current_hp=hp,
            max_hp=100,
            current_hp_fraction=hp / 100,
            boosts=boosts or {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
            moves={m.id: m for m in (moves or [])},
            fainted=False,
            active=False,
        )

    def test_passive_turn_against_faster_special_attacker_activates_defensive_switch(self):
        leech_seed = FakeMove("leechseed")
        psychic = FakeMove("psychic", 90)
        celebi = self._pokemon(
            "celebi",
            types=("Grass", "Psychic"),
            base_stats={"hp": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100},
            boosts={"atk": 0, "def": 0, "spa": 1, "spd": 1, "spe": 0},
            moves=[leech_seed, psychic],
        )
        blissey = self._pokemon(
            "blissey",
            types=("Normal",),
            base_stats={"hp": 255, "def": 10, "spa": 75, "spd": 135, "spe": 55},
            hp=100,
        )
        starmie = self._pokemon(
            "starmie",
            types=("Water", "Psychic"),
            base_stats={"hp": 60, "def": 85, "spa": 100, "spd": 85, "spe": 115},
            moves=[],
        )
        celebi.active = True
        battle = SimpleNamespace(
            active_pokemon=celebi,
            opponent_active_pokemon=starmie,
            team={"celebi": celebi, "blissey": blissey},
        )

        action, reason = hidden_threat_switch_override(
            battle,
            [1, 4],
            1,
            {0: leech_seed, 1: psychic},
        )

        self.assertEqual(action, 4)
        self.assertIn("hidden-threat response", reason)
        self.assertIn("blissey", reason)

    def test_damaging_turn_is_not_overridden(self):
        psychic = FakeMove("psychic", 90)
        celebi = self._pokemon(
            "celebi",
            types=("Grass", "Psychic"),
            base_stats={"hp": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100},
            moves=[psychic],
        )
        blissey = self._pokemon(
            "blissey",
            base_stats={"hp": 255, "def": 10, "spa": 75, "spd": 135, "spe": 55},
        )
        starmie = self._pokemon(
            "starmie",
            types=("Water", "Psychic"),
            base_stats={"hp": 60, "def": 85, "spa": 100, "spd": 85, "spe": 115},
        )
        celebi.active = True
        battle = SimpleNamespace(
            active_pokemon=celebi,
            opponent_active_pokemon=starmie,
            team={"celebi": celebi, "blissey": blissey},
        )

        action, reason = hidden_threat_switch_override(
            battle,
            [0, 4],
            0,
            {0: psychic},
        )

        self.assertIsNone(action)
        self.assertEqual(reason, "")


if __name__ == "__main__":
    unittest.main()
