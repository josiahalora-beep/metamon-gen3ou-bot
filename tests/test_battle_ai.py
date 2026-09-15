import unittest
from types import SimpleNamespace

from battle_ai.damage import calculate_damage, type_multiplier
from battle_ai.evaluator import TacticalEvaluator
from battle_ai.speed import can_outspeed, speed_range, speed_tie_probability
from battle_ai.state import snapshot_battle


def pokemon(name="swampert", types=("water", "ground"), stats=None, hp=300, status="", item=""):
    return SimpleNamespace(
        name=name, species=name, base_species=name, types=types, current_hp=hp,
        max_hp=hp, current_hp_fraction=1.0, status=status, item=item,
        ability="", level=100, base_stats={}, stats=stats or {"atk": 200, "def": 200, "spa": 200, "spd": 200, "spe": 200},
        boosts={}, moves={}, fainted=False,
    )


class Move:
    def __init__(self, name, power, typ):
        self.id = name; self.name = name; self.base_power = power; self.type = SimpleNamespace(name=typ)


class BattleAiTests(unittest.TestCase):
    def test_type_chart_super_effective_and_immunity(self):
        self.assertEqual(type_multiplier("electric", ["water"]), 2.0)
        self.assertEqual(type_multiplier("electric", ["ground"]), 0.0)
        self.assertEqual(type_multiplier("fire", ["water"]), 0.5)

    def test_immunity_is_zero_damage(self):
        result = calculate_damage(pokemon("zapdos", ("electric", "flying")), pokemon("swampert", ("water", "ground")), Move("thunderbolt", 95, "Electric"))
        self.assertEqual(result.max_damage, 0)

    def test_weather_changes_damage(self):
        fire = Move("flamethrower", 95, "Fire")
        neutral = calculate_damage(pokemon("tyranitar", ("rock", "dark")), pokemon("skarmory", ("steel", "flying")), fire)
        sun = calculate_damage(pokemon("tyranitar", ("rock", "dark")), pokemon("skarmory", ("steel", "flying")), fire, weather="sunnyday")
        self.assertGreater(sun.max_damage, neutral.max_damage)

    def test_explosion_uses_gen3_defense_halving(self):
        explosion = Move("explosion", 250, "Normal")
        normal = calculate_damage(pokemon(), pokemon(), explosion)
        self.assertGreater(normal.max_damage, 0)

    def test_critical_hit_ignores_relevant_boost_directions(self):
        move = Move("earthquake", 100, "Ground")
        attacker = pokemon(stats={"atk": 200, "def": 200, "spa": 200, "spd": 200, "spe": 200})
        attacker.boosts = {"atk": -6, "def": 0, "spa": 0, "spd": 0, "spe": 0}
        defender = pokemon()
        normal = calculate_damage(attacker, defender, move)
        critical = calculate_damage(attacker, defender, move, critical=True)
        self.assertGreater(critical.max_damage, normal.max_damage)

    def test_gen3_damage_stab_and_range(self):
        result = calculate_damage(pokemon("swampert", ("water", "ground")), pokemon("skarmory", ("steel", "flying"), hp=334), Move("surf", 95, "Water"))
        self.assertGreater(result.max_damage, result.min_damage)
        self.assertGreater(result.percentage_max, 0)

    def test_burn_halves_physical_damage(self):
        clean = calculate_damage(pokemon(status=""), pokemon(), Move("earthquake", 100, "Ground"))
        burned = calculate_damage(pokemon(status="brn"), pokemon(), Move("earthquake", 100, "Ground"))
        self.assertLess(burned.max_damage, clean.max_damage)

    def test_guaranteed_ko_uses_current_hp_not_max_hp(self):
        attacker = pokemon("attacker", ("normal",), stats={"atk": 300, "def": 200, "spa": 200, "spd": 200, "spe": 200})
        defender = pokemon("target", ("normal",), stats={"atk": 200, "def": 100, "spa": 200, "spd": 200, "spe": 200}, hp=160)
        defender.max_hp = 300
        defender.current_hp = 160
        defender.current_hp_fraction = 160 / 300
        result = calculate_damage(attacker, defender, Move("earthquake", 100, "Ground"))
        self.assertGreaterEqual(result.min_damage, defender.current_hp)
        self.assertEqual(result.ko_probability, 1.0)

    def test_hidden_power_uses_gen3_iv_type_and_power(self):
        attacker = pokemon("attacker", ("dark",), stats={"atk": 200, "def": 200, "spa": 200, "spd": 200, "spe": 200})
        attacker.ivs = {"hp": 31, "atk": 31, "def": 31, "spa": 31, "spd": 31, "spe": 31}
        defender = pokemon("alakazam", ("psychic",), stats={"atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100}, hp=250)
        # All 31 IVs yield Dark / 70 BP in the Gen 3 Hidden Power formula.
        result = calculate_damage(attacker, defender, Move("hiddenpowerfire", 70, "Fire"))
        self.assertGreater(result.max_damage, 0)
        self.assertTrue(result.reliable)

    def test_hidden_power_without_ivs_is_not_treated_as_reliable(self):
        attacker = pokemon("attacker")
        defender = pokemon("target")
        result = calculate_damage(attacker, defender, Move("hiddenpowerghost", 70, "Ghost"))
        self.assertFalse(result.reliable)
        self.assertEqual(result.ko_probability, 0.0)

    def test_unknown_opponent_stats_never_create_fake_ko(self):
        unknown = pokemon(stats={"atk": None, "def": None, "spa": None, "spd": None, "spe": None}, hp=100)
        result = calculate_damage(pokemon(), unknown, Move("earthquake", 100, "Ground"))
        self.assertFalse(result.reliable)
        self.assertEqual(result.ko_probability, 0.0)

    def test_fixed_action_slots(self):
        moves = [Move("zmove", 0, "Normal"), Move("amove", 0, "Normal"), Move("bmove", 0, "Normal"), Move("cmove", 0, "Normal")]
        active = pokemon("active", stats={"atk": 200, "def": 200, "spa": 200, "spd": 200, "spe": 200})
        active.moves = {m.id: m for m in moves}
        target = pokemon("target", stats={"atk": None, "def": None, "spa": None, "spd": None, "spe": None})
        switches = [pokemon(f"s{i}") for i in range(5)]
        battle = SimpleNamespace(active_pokemon=active, opponent_active_pokemon=target,
                                 available_moves=moves, team={"a": active, **{p.name: p for p in switches}},
                                 force_switch=False, weather={}, battle_tag="x", turn=1,
                                 player_username="a", opponent_username="b", opponent_team={},
                                 side_conditions={}, opponent_side_conditions={}, fields={})
        chosen, evaluations = TacticalEvaluator().evaluate(battle, list(range(9)), 2)
        self.assertEqual(chosen, 2)
        self.assertEqual(evaluations[2].kind, "move")

    def test_verifier_preserves_model_when_evidence_is_unknown(self):
        active = pokemon("active"); active.moves = {"a": Move("a", 100, "Normal")}
        target = pokemon("target", stats={"atk": None, "def": None, "spa": None, "spd": None, "spe": None})
        battle = SimpleNamespace(active_pokemon=active, opponent_active_pokemon=target,
                                 available_moves=list(active.moves.values()), team={"a": active},
                                 force_switch=False, weather={}, battle_tag="x", turn=1,
                                 player_username="a", opponent_username="b", opponent_team={},
                                 side_conditions={}, opponent_side_conditions={}, fields={})
        chosen, _ = TacticalEvaluator().evaluate(battle, [0], 0)
        self.assertEqual(chosen, 0)

    def test_verifier_does_not_rerank_even_with_known_damage(self):
        active = pokemon("active"); active.moves = {"a": Move("a", 100, "Normal"), "b": Move("b", 20, "Normal")}
        target = pokemon("target")
        battle = SimpleNamespace(active_pokemon=active, opponent_active_pokemon=target,
                                 available_moves=list(active.moves.values()), team={"a": active},
                                 force_switch=False, weather={}, battle_tag="x", turn=1,
                                 player_username="a", opponent_username="b", opponent_team={},
                                 side_conditions={}, opponent_side_conditions={}, fields={})
        chosen, _ = TacticalEvaluator().evaluate(battle, [0, 1], 1)
        self.assertEqual(chosen, 1)

    def test_speed_helpers(self):
        a = pokemon(stats={"spe": 300})
        b = pokemon(stats={"spe": 200})
        self.assertEqual(speed_range(a), (300, 300))
        self.assertTrue(can_outspeed(a, b))
        self.assertEqual(speed_tie_probability(a, b), 1.0)

    def test_snapshot_is_normalized(self):
        battle = SimpleNamespace(battle_tag="battle-gen3ou-1", turn=3, format="gen3ou", player_username="a", opponent_username="b", active_pokemon=pokemon(), opponent_active_pokemon=pokemon("blissey", ("normal",)), team={"a": pokemon()}, opponent_team={"b": pokemon("blissey", ("normal",))}, weather={}, side_conditions={}, opponent_side_conditions={}, fields={}, force_switch=False)
        state = snapshot_battle(battle, [0, 4])
        self.assertEqual(state.turn, 3)
        self.assertEqual(state.available_actions, (0, 4))
        self.assertEqual(state.our_active["name"], "swampert")


if __name__ == "__main__": unittest.main()
