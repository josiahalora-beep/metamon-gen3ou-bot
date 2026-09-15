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


class PredictiveSafetyRegressionTests(unittest.TestCase):
    @staticmethod
    def pokemon(name, *, hp=100, moves=None, types=("Normal",), active=False):
        return SimpleNamespace(
            species=name,
            name=name,
            current_hp=hp,
            max_hp=100,
            current_hp_fraction=hp / 100,
            moves={m.id: m for m in (moves or [])},
            types=types,
            base_stats={"hp": 100, "atk": 120, "def": 100, "spa": 100, "spd": 100, "spe": 100},
            boosts={"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
            active=active,
            fainted=False,
            level=100,
            stats={"atk": 236, "def": 236, "spa": 236, "spd": 236, "spe": 236},
        )

    def test_model_switch_cannot_become_explosion_trade(self):
        model = OpponentModel()
        model._profile_move_prior = lambda pokemon: {"flamethrower": 1.0}
        search = ResponseSearcher(model, override_margin=0.1)
        explosion = FakeMove("explosion", 250, "Normal")
        switch_fodder = self.pokemon("Salamence", active=True)
        opp = self.pokemon("Moltres", moves=[FakeMove("flamethrower", 95, "Fire")], types=("Fire", "Flying"), active=True)
        teammate = self.pokemon("Suicune", types=("Water",), active=False)
        battle = SimpleNamespace(
            battle_tag="reg-explosion",
            active_pokemon=switch_fodder,
            opponent_active_pokemon=opp,
            team={"Salamence": switch_fodder, "Suicune": teammate},
            opponent_team={"Moltres": opp},
        )
        # Search must not manufacture a self-KO override when the learned policy
        # already chose the defensive switch action 4.
        action, _, _ = search.choose(battle, [0, 4], 4)
        self.assertIsNone(action)

    def test_setup_move_is_not_replaced_by_predicted_attack(self):
        model = OpponentModel()
        model._profile_move_prior = lambda pokemon: {"earthquake": 1.0}
        search = ResponseSearcher(model, override_margin=0.1)
        setup = FakeMove("swordsdance", 0, "Normal")
        attack = FakeMove("rockslide", 75, "Rock")
        own = self.pokemon("Heracross", moves=[setup, attack], types=("Bug", "Fighting"), active=True)
        opp = self.pokemon("Metagross", moves=[FakeMove("earthquake", 100, "Ground")], types=("Steel", "Psychic"), active=True)
        battle = SimpleNamespace(
            battle_tag="reg-setup",
            active_pokemon=own,
            opponent_active_pokemon=opp,
            team={"Heracross": own},
            opponent_team={"Metagross": opp},
        )
        action, _, _ = search.choose(battle, [0, 1], 0)
        self.assertIsNone(action)


if __name__ == "__main__":
    unittest.main()
