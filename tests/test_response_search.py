import unittest
from types import SimpleNamespace

from battle_ai.opponent_model import OpponentModel
from battle_ai.response_search import ResponseSearcher


class FakeMove:
    def __init__(self, move_id, power=0, move_type="Normal"):
        self.id = move_id
        self.name = move_id
        self.base_power = power
        self.type = move_type


class ResponseSearchTests(unittest.TestCase):
    @staticmethod
    def pokemon(name, *, hp=100, moves=None, types=("Normal",), base_stats=None, active=False):
        return SimpleNamespace(
            species=name,
            name=name,
            current_hp=hp,
            max_hp=100,
            current_hp_fraction=hp / 100,
            moves={m.id: m for m in (moves or [])},
            types=types,
            base_stats=base_stats or {"hp": 100, "atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100},
            boosts={"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
            active=active,
            fainted=False,
            level=100,
            stats={"atk": 236, "def": 236, "spa": 236, "spd": 236, "spe": 236},
        )

    def test_model_action_survives_when_search_advantage_is_small(self):
        model = OpponentModel()
        model._profile_move_prior = lambda pokemon: {"earthquake": 1.0}
        search = ResponseSearcher(model, override_margin=2.0)
        our_move = FakeMove("surf", 95, "Water")
        own = self.pokemon("Swampert", moves=[our_move], types=("Water", "Ground"), active=True)
        opp_move = FakeMove("earthquake", 100, "Ground")
        opp = self.pokemon("Flygon", moves=[opp_move], types=("Ground", "Dragon"), active=True)
        battle = SimpleNamespace(
            battle_tag="battle-small",
            active_pokemon=own,
            opponent_active_pokemon=opp,
            team={"Swampert": own},
            opponent_team={"Flygon": opp},
        )
        action, reason, scores = search.choose(battle, [0], 0)
        self.assertIsNone(action)
        self.assertEqual(reason, "")
        self.assertEqual(scores[0].action, 0)

    def test_search_can_predict_a_switch_punishment(self):
        model = OpponentModel()
        model._profile_move_prior = lambda pokemon: {"earthquake": 1.0}
        search = ResponseSearcher(model, override_margin=0.5)
        surf = FakeMove("surf", 95, "Water")
        icebeam = FakeMove("icebeam", 95, "Ice")
        own = self.pokemon("Starmie", moves=[surf, icebeam], types=("Water", "Psychic"), active=True,
                           base_stats={"hp": 60, "atk": 75, "def": 85, "spa": 100, "spd": 85, "spe": 115})
        opp_move = FakeMove("earthquake", 100, "Ground")
        opp = self.pokemon("Flygon", moves=[opp_move], types=("Ground", "Dragon"), active=True)
        incoming = self.pokemon("Swampert", types=("Water", "Ground"), active=False,
                                 base_stats={"hp": 100, "atk": 110, "def": 90, "spa": 85, "spd": 90, "spe": 60})
        battle = SimpleNamespace(
            battle_tag="battle-switch",
            active_pokemon=own,
            opponent_active_pokemon=opp,
            team={"Starmie": own},
            opponent_team={"Flygon": opp, "Swampert": incoming},
        )
        responses = model.predict_responses(battle)
        self.assertTrue(any(r.kind == "switch" for r in responses))
        action, reason, scores = search.choose(battle, [0, 1], 0)
        self.assertIn(action, {None, 1})
        if action == 1:
            self.assertIn("response search", reason)

    def test_predictive_search_never_picks_immune_move_without_strong_switch_evidence(self):
        model = OpponentModel()
        model._profile_move_prior = lambda pokemon: {"thunderbolt": 1.0}
        search = ResponseSearcher(model, override_margin=0.1)
        earthquake = FakeMove("earthquake", 100, "Ground")
        rockslide = FakeMove("rockslide", 75, "Rock")
        own = self.pokemon("Tyranitar", moves=[earthquake, rockslide], types=("Rock", "Dark"), active=True,
                           base_stats={"hp": 100, "atk": 134, "def": 110, "spa": 95, "spd": 100, "spe": 61})
        thunderbolt = FakeMove("thunderbolt", 95, "Electric")
        opp = self.pokemon("Zapdos", moves=[thunderbolt], types=("Electric", "Flying"), active=True,
                           base_stats={"hp": 90, "atk": 90, "def": 85, "spa": 125, "spd": 90, "spe": 100})
        battle = SimpleNamespace(
            battle_tag="battle-immunity",
            active_pokemon=own,
            opponent_active_pokemon=opp,
            team={"Tyranitar": own},
            opponent_team={"Zapdos": opp},
        )
        action, reason, scores = search.choose(battle, [0, 1], 0)
        self.assertNotEqual(action, 0)
        self.assertNotIn("best=0", reason)


if __name__ == "__main__":
    unittest.main()
