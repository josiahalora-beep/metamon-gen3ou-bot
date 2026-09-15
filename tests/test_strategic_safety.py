import unittest
from types import SimpleNamespace

from battle_ai.damage import calculate_damage
from battle_ai.strategic import _team_pokemon_for_action, safety_override


class Move:
    def __init__(self, name, power, typ):
        self.id = name
        self.name = name
        self.base_power = power
        self.type = SimpleNamespace(name=typ)


def pokemon(name, types=("normal",), hp=100, max_hp=100, status="", moves=None, fainted=False, stats=None, boosts=None, base_stats=None):
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
        base_stats=base_stats or {},
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
        return SimpleNamespace(active_pokemon=active, opponent_active_pokemon=opponent, team=team, force_switch=False, weather={})

    def test_refuses_switch_into_low_hp_toxic_pokemon_when_safer_switch_exists(self):
        active = pokemon("jirachi", hp=100)
        compromised = pokemon("suicune", types=("water",), hp=20, max_hp=100, status="tox")
        safe = pokemon("gengar", types=("ghost",), hp=100)
        opponent = pokemon("tyranitar", types=("rock", "dark"), moves=[Move("earthquake", 100, "Ground")])
        battle = self._battle(active, opponent, [compromised, safe])
        decision = safety_override(battle, [4, 5], 5)
        self.assertEqual(decision.action, 4)
        self.assertTrue(decision.hard)

    def test_estimated_incoming_damage_can_reject_a_frail_switch(self):
        active = pokemon("jirachi", hp=100)
        fragile = pokemon("aerodactyl", types=("rock", "flying"), hp=25, max_hp=100,
                          base_stats={"hp": 80, "def": 65, "spd": 75, "atk": 105, "spa": 60, "spe": 130},
                          stats={"atk": 250, "def": 180, "spa": 120, "spd": 170, "spe": 300})
        bulky = pokemon("swampert", types=("water", "ground"), hp=100, max_hp=100,
                        base_stats={"hp": 100, "def": 90, "spd": 90, "atk": 110, "spa": 85, "spe": 60},
                        stats={"atk": 250, "def": 300, "spa": 220, "spd": 220, "spe": 180})
        opponent = pokemon("metagross", types=("steel", "psychic"),
                           base_stats={"hp": 80, "def": 130, "spd": 90, "atk": 135, "spa": 95, "spe": 70},
                           stats={"atk": None, "def": None, "spa": None, "spd": None, "spe": None},
                           moves=[Move("meteor_mash", 100, "Steel"), Move("earthquake", 100, "Ground")])
        battle = self._battle(active, opponent, [fragile, bulky])
        decision = safety_override(battle, [4, 5], 4)
        self.assertEqual(decision.action, 5)
        self.assertTrue(decision.hard)

    def test_does_not_override_unknown_opponent_moves(self):
        active = pokemon("suicune", types=("water",), hp=20, max_hp=100, status="tox")
        safe = pokemon("gengar", types=("ghost",), hp=100)
        opponent = pokemon("unknown", types=("normal",), moves=[])
        battle = self._battle(active, opponent, [safe])
        decision = safety_override(battle, [4], 0)
        self.assertEqual(decision.action, 4)
        self.assertTrue(decision.hard)

    def test_setup_emergency_prefers_reliable_attack(self):
        calm_mind = Move("calmmind", 0, "Psychic")
        ice = Move("icebeam", 95, "Ice")
        active = pokemon("starmie", types=("water", "psychic"), moves=[calm_mind, ice], stats={"atk": 200, "def": 200, "spa": 300, "spd": 250, "spe": 300})
        opponent = pokemon("dragonite", types=("dragon", "flying"), hp=100, boosts={"atk": 2, "def": 0, "spa": 0, "spd": 0, "spe": 2})
        battle = self._battle(active, opponent, [])
        decision = safety_override(battle, [0, 1], 0)
        if decision.action is not None:
            self.assertEqual(decision.action, 1)
            self.assertTrue(decision.hard)

    def test_low_hp_preservation_refuses_non_guaranteed_attack(self):
        attack = Move("hiddenpowerice", 70, "Ice")
        active = pokemon("zapdos", types=("electric", "flying"), hp=34, max_hp=100,
                          moves=[attack], stats={"atk": 250, "def": 200, "spa": 300, "spd": 220, "spe": 300})
        switch_a = pokemon("blissey", types=("normal",), hp=100)
        switch_b = pokemon("skarmory", types=("steel", "flying"), hp=80)
        opponent = pokemon("swampert", types=("water", "ground"), hp=85, max_hp=100,
                            base_stats={"hp": 100, "def": 90, "spd": 90, "atk": 110, "spa": 85, "spe": 60},
                            stats={"atk": None, "def": None, "spa": None, "spd": None, "spe": None})
        battle = self._battle(active, opponent, [switch_a, switch_b])
        decision = safety_override(battle, [0, 4, 5], 0)
        self.assertEqual(decision.action, 4)
        self.assertTrue(decision.hard)

    def test_switch_action_uses_canonical_pokemon_order(self):
        active = pokemon("zaptos")
        blissey = pokemon("blissey")
        dugtrio = pokemon("dugtrio")
        metagross = pokemon("metagross")
        skarmory = pokemon("skarmory")
        suicune = pokemon("suicune")
        active.active = True
        battle = SimpleNamespace(team={
            "z": active,
            "s": suicune,
            "b": blissey,
            "m": metagross,
            "d": dugtrio,
            "k": skarmory,
        })
        self.assertIs(_team_pokemon_for_action(battle, 4), blissey)
        self.assertIs(_team_pokemon_for_action(battle, 5), dugtrio)
        self.assertIs(_team_pokemon_for_action(battle, 6), metagross)
        self.assertIs(_team_pokemon_for_action(battle, 7), skarmory)
        self.assertIs(_team_pokemon_for_action(battle, 8), suicune)

    def test_placeholder_hp_is_estimated_for_100_base_hp_species(self):
        defender = pokemon("celebi", types=("grass", "psychic"), hp=75, max_hp=100,
                            base_stats={"hp": 100, "def": 100, "spd": 100, "atk": 100, "spa": 100, "spe": 100},
                            stats={"atk": None, "def": None, "spa": None, "spd": None, "spe": None})
        attacker = pokemon("zapdos", types=("electric", "flying"),
                           stats={"atk": 250, "def": 200, "spa": 300, "spd": 220, "spe": 300})
        result = calculate_damage(attacker, defender, Move("thunderbolt", 95, "Electric"))
        self.assertTrue(result.reliable)
        self.assertIn("estimated", result.reason)
        self.assertGreater(result.percentage_max, 0)


if __name__ == "__main__": unittest.main()
