import unittest
from types import SimpleNamespace
from unittest.mock import patch

from battle_ai.strategic_plan import strategic_opportunity_override


class FakeMove:
    def __init__(self, move_id, power=0, move_type="Normal", *, protect=False):
        self.id = move_id
        self.name = move_id
        self.base_power = power
        self.type = SimpleNamespace(name=move_type)
        self.is_protect_counter = protect


class FakeResult:
    def __init__(self, *, ko_probability=0.0, reliable=True, max_damage=20.0):
        self.ko_probability = ko_probability
        self.reliable = reliable
        self.max_damage = max_damage
        self.percentage_max = max_damage
        self.percentage_min = max_damage / 2
        self.reason = ""


class StrategicPlanTests(unittest.TestCase):
    def _pokemon(self, name, moves=None, hp=100, max_hp=100, types=("Normal",), boosts=None, ability=""):
        return SimpleNamespace(
            name=name,
            species=name,
            types=types,
            current_hp=hp,
            max_hp=max_hp,
            current_hp_fraction=hp / max_hp,
            status="",
            ability=ability,
            boosts=boosts or {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
            moves={m.id: m for m in (moves or [])},
            fainted=False,
            active=False,
        )

    def _battle(self, active, opponent, opponent_team, *, team=None, hazards=None, legal=(0, 1)):
        active.active = True
        if team is None:
            team = {"active": active}
        return SimpleNamespace(
            active_pokemon=active,
            opponent_active_pokemon=opponent,
            opponent_team={str(i): p for i, p in enumerate(opponent_team)},
            team=team,
            available_moves=list(active.moves.values()),
            opponent_side_conditions=hazards or {},
            weather={},
            force_switch=False,
        )

    def test_setup_move_is_chosen_when_it_creates_a_real_sweep_window(self):
        dd = FakeMove("dragondance")
        attack = FakeMove("return", 100)
        active = self._pokemon("mence", moves=[dd, attack])
        foe1 = self._pokemon("skarmory", types=("Steel", "Flying"))
        foe2 = self._pokemon("blissey", types=("Normal",))
        battle = self._battle(active, foe1, [foe1, foe2])

        def fake_damage(attacker, defender, move, weather=""):
            if move.id != "return":
                return FakeResult(ko_probability=0.0, max_damage=0.0)
            boosted = int((getattr(attacker, "boosts", {}) or {}).get("atk", 0)) >= 1
            return FakeResult(ko_probability=1.0 if boosted else 0.0, max_damage=100.0 if boosted else 45.0)

        with patch("battle_ai.strategic_plan.calculate_damage", side_effect=fake_damage):
            with patch("battle_ai.strategic_plan._incoming_safe", return_value=(True, 20.0, "known attack")):
                action, reason = strategic_opportunity_override(battle, [0, 1], 1)

        self.assertEqual(action, 0)
        self.assertIn("sweep window", reason)

    def test_spikes_are_stacked_when_two_grounded_targets_remain(self):
        spikes = FakeMove("spikes")
        quake = FakeMove("earthquake", 100, "Ground")
        active = self._pokemon("forretress", moves=[spikes, quake], types=("Bug", "Steel"))
        foe1 = self._pokemon("metagross", types=("Steel", "Psychic"))
        foe2 = self._pokemon("tyranitar", types=("Rock", "Dark"))
        battle = self._battle(active, foe1, [foe1, foe2], hazards={"spikes": 1})

        def fake_damage(attacker, defender, move, weather=""):
            return FakeResult(ko_probability=0.0, max_damage=25.0)

        with patch("battle_ai.strategic_plan.calculate_damage", side_effect=fake_damage):
            with patch("battle_ai.strategic_plan._incoming_safe", return_value=(True, 20.0, "known attack")):
                action, reason = strategic_opportunity_override(battle, [0, 1], 1)

        self.assertEqual(action, 1)
        self.assertIn("Spikes layer 2/3", reason)

    def test_does_not_stack_spikes_when_the_turn_is_not_safe(self):
        spikes = FakeMove("spikes")
        quake = FakeMove("earthquake", 100, "Ground")
        active = self._pokemon("forretress", moves=[spikes, quake], types=("Bug", "Steel"))
        foe1 = self._pokemon("medicham", types=("Fighting", "Psychic"))
        foe2 = self._pokemon("tyranitar", types=("Rock", "Dark"))
        battle = self._battle(active, foe1, [foe1, foe2], hazards={"spikes": 1})

        with patch("battle_ai.strategic_plan._incoming_safe", return_value=(False, 80.0, "hiddenpower")):
            action, reason = strategic_opportunity_override(battle, [0, 1], 0)

        self.assertIsNone(action)
        self.assertEqual(reason, "")

    def test_boosted_primary_win_condition_executes_guaranteed_ko(self):
        curse = FakeMove("curse")
        return_move = FakeMove("return", 100, "Normal")
        active = self._pokemon(
            "snorlax",
            moves=[curse, return_move],
            boosts={"atk": 1, "def": 1, "spa": 0, "spd": 0, "spe": -1},
        )
        foe = self._pokemon("tyranitar", types=("Rock", "Dark"))
        battle = self._battle(active, foe, [foe])

        def fake_damage(attacker, defender, move, weather=""):
            if move.id == "return" and int((getattr(attacker, "boosts", {}) or {}).get("atk", 0)) >= 1:
                return FakeResult(ko_probability=1.0, max_damage=120.0)
            if move.id == "return":
                return FakeResult(ko_probability=0.0, max_damage=60.0)
            return FakeResult(ko_probability=0.0, max_damage=0.0)

        with patch("battle_ai.strategic_plan.calculate_damage", side_effect=fake_damage):
            with patch("battle_ai.strategic_plan._incoming_safe", return_value=(True, 10.0, "known attack")):
                action, reason = strategic_opportunity_override(battle, [0, 1], 0)

        self.assertEqual(action, 1)
        self.assertIn("cleanup execution", reason)
        self.assertIn("guaranteed KO", reason)

    def test_switch_to_primary_win_condition_when_it_is_a_safe_sweep_activation(self):
        current_attack = FakeMove("surf", 95, "Water")
        current = self._pokemon("suicune", moves=[current_attack])
        dd = FakeMove("dragondance")
        dragon_attack = FakeMove("dragonclaw", 80, "Dragon")
        mence = self._pokemon("mence", moves=[dd, dragon_attack], types=("Dragon", "Flying"), hp=100)
        foe1 = self._pokemon("blissey", types=("Normal",))
        foe2 = self._pokemon("metagross", types=("Steel", "Psychic"))
        team = {"current": current, "mence": mence}
        battle = self._battle(current, foe1, [foe1, foe2], team=team)
        battle.available_moves = list(current.moves.values())

        def fake_damage(attacker, defender, move, weather=""):
            if move.id == "dragonclaw":
                boosted = int((getattr(attacker, "boosts", {}) or {}).get("atk", 0)) >= 1
                return FakeResult(ko_probability=1.0 if boosted else 0.0, max_damage=100.0 if boosted else 40.0)
            return FakeResult(ko_probability=0.0, max_damage=20.0)

        with patch("battle_ai.strategic_plan.calculate_damage", side_effect=fake_damage):
            with patch("battle_ai.strategic_plan._incoming_safe", return_value=(True, 10.0, "known attack")):
                action, reason = strategic_opportunity_override(battle, [0, 4], 4)

        # Current fixture uses consistent alphabetical switch ordering: mence is action 4.
        self.assertEqual(action, 4)
        self.assertIn("win-condition activation", reason)


if __name__ == "__main__":
    unittest.main()
