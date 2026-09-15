import unittest
from types import SimpleNamespace

from battle_ai.opponent_model import OpponentModel


class FakeMove:
    def __init__(self, move_id, power=0):
        self.id = move_id
        self.name = move_id
        self.base_power = power
        self.type = "Normal"


class OpponentModelTests(unittest.TestCase):
    @staticmethod
    def pokemon(name, *, hp=100, moves=None, base_stats=None, types=("Normal",), active=False, fainted=False):
        return SimpleNamespace(
            species=name,
            name=name,
            current_hp_fraction=hp / 100,
            moves={m.id: m for m in (moves or [])},
            base_stats=base_stats or {"hp": 80, "atk": 80, "def": 80, "spa": 80, "spd": 80, "spe": 80},
            types=types,
            active=active,
            fainted=fainted,
        )

    def test_switch_observation_is_smoothed_not_overfit(self):
        model = OpponentModel(prior_strength=8.0)
        first = self.pokemon("Tyranitar", active=True)
        second = self.pokemon("Swampert", active=True)
        battle = SimpleNamespace(
            battle_tag="battle-1",
            active_pokemon=self.pokemon("Celebi", active=True),
            opponent_active_pokemon=first,
            opponent_team={"Tyranitar": first, "Swampert": second},
        )
        model.observe_transition(battle)
        battle.opponent_active_pokemon = second
        model.observe_transition(battle)
        state = model._behavior["battle-1"]
        self.assertEqual(state.switches, 1)
        self.assertEqual(state.switch_opportunities, 1)
        self.assertGreater(state.smoothed_switch_rate(), 0.3)
        self.assertLess(state.smoothed_switch_rate(), 0.9)

    def test_predict_responses_retains_unknown_probability_mass(self):
        model = OpponentModel()
        model._profile_move_prior = lambda pokemon: {"earthquake": 0.7, "protect": 0.3}
        opp = self.pokemon("Tyranitar", active=True, moves=[FakeMove("earthquake", 100)])
        own = self.pokemon("Metagross", active=True)
        bench = self.pokemon("Skarmory", active=False, types=("Steel", "Flying"), hp=100)
        battle = SimpleNamespace(
            battle_tag="battle-2",
            active_pokemon=own,
            opponent_active_pokemon=opp,
            opponent_team={"Tyranitar": opp, "Skarmory": bench},
        )
        responses = model.predict_responses(battle)
        self.assertTrue(responses)
        self.assertLessEqual(sum(r.probability for r in responses), 1.001)
        self.assertGreaterEqual(sum(r.probability for r in responses), 0.999)
        self.assertTrue(any(r.kind == "switch" for r in responses))

    def test_protect_and_setup_are_distinct_behavior_classes(self):
        model = OpponentModel()
        model._profile_move_prior = lambda pokemon: {
            "protect": 0.4,
            "swordsdance": 0.3,
            "earthquake": 0.3,
        }
        opp = self.pokemon("Garchomp", active=True)
        own = self.pokemon("Swampert", active=True)
        battle = SimpleNamespace(
            battle_tag="battle-3",
            active_pokemon=own,
            opponent_active_pokemon=opp,
            opponent_team={"Garchomp": opp},
        )
        responses = model.predict_responses(battle)
        kinds = {r.kind for r in responses}
        self.assertIn("protect", kinds)
        self.assertIn("setup", kinds)
        self.assertIn("attack", kinds)


if __name__ == "__main__":
    unittest.main()
