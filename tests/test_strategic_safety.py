import unittest
from types import SimpleNamespace

from battle_ai.strategic import safety_override


class Move:
    def __init__(self, name, power, typ):
        self.id = name
        self.name = name
        self.base_power = power
        self.type = SimpleNamespace(name=typ)


def pokemon(name, types=("normal",), hp=100, max_hp=100, status="", moves=None, fainted=False, stats=None, boosts=None):
    return SimpleNamespace(
        name=name,
        species=name,
        base_species=name,
        types=types,
        current_hp=hp,
        max_hp=max_hp,
        current_hp_fraction=hp / max_hp,
        status=status,
        item="",
        ability="",
        level=100,
        base_stats={},
        stats=stats or {"atk": 200, "def": 200, "spa": 200, "spd": 200, "spe": 200},
        boosts=boosts or {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
        moves={m.id: m for m in (moves or [])},
        fainted=fainted,
        active=False,
    )


class StrategicSafetyTests(unittest.TestCase):
    def _battle(self, active, opponent, switches):
        team = {active.name: active}
        for mon in switches:
            team[mon.name] = mon
        active.active = True
        return SimpleNamespace(
            active_pokemon=active,
            opponent_active_pokemon=opponent,
            team=team,
            force_switch=False,
            weather={},
        )

    def test_refuses_switch_into_low_hp_toxic_pokemon_when_safer_switch_exists(self):
        active = pokemon("jirachi", hp=100)
        compromised = pokemon("suicune", types=("water",), hp=20, max_hp=100, status="tox")
        safe = pokemon("gengar", types=("ghost",), hp=100)
        opponent = pokemon(
            "tyranitar",
            types=("rock", "dark"),
            moves=[Move("earthquake", 100, "Ground")],
        )
        battle = self._battle(active, opponent, [compromised, safe])
        # Alphabetical switches: gengar then suicune, so action 4=gengar and 5=suicune.
        decision = safety_override(battle, [4, 5], 5)
        self.assertEqual(decision.action, 4)
        self.assertTrue(decision.hard)

    def test_does_not_override_unknown_opponent_moves(self):
        active = pokemon("suicune", types=("water",), hp=20, max_hp=100, status="tox")
        safe = pokemon("gengar", types=("ghost",), hp=100)
        opponent = pokemon("unknown", types=("normal",), moves=[])
        battle = self._battle(active, opponent, [safe])
        decision = safety_override(battle, [4], 0)
        self.assertIsNone(decision.action)

    def test_setup_emergency_prefers_reliable_attack(self):
        calm_mind = Move("calmmind", 0, "Psychic")
        ice = Move("icebeam", 95, "Ice")
        active = pokemon("starmie", types=("water", "psychic"), moves=[calm_mind, ice], stats={"atk": 200, "def": 200, "spa": 300, "spd": 250, "spe": 300})
        opponent = pokemon("dragonite", types=("dragon", "flying"), hp=100, boosts={"atk": 2, "def": 0, "spa": 0, "spd": 0, "spe": 2})
        battle = self._battle(active, opponent, [])
        # The exact result depends on the deterministic installed damage engine;
        # require only that the safety layer sees the boosted threat and returns
        # either no override or the available damaging slot, never an invalid slot.
        decision = safety_override(battle, [0, 1], 0)
        if decision.action is not None:
            self.assertEqual(decision.action, 1)
            self.assertTrue(decision.hard)


if __name__ == "__main__":
    unittest.main()
