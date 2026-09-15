import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from battle_ai.override_gate import hard_loss_switch_allowed


class OverrideGateTests(unittest.TestCase):
    def _battle(self, *, incoming_ko=True, selected_ko=False, candidate_ko=False):
        move = NS(id="earthquake", name="earthquake", base_power=100)
        incoming = NS(id="surf", name="surf", base_power=95)
        active = NS(species="metagross", active=True, fainted=False, moves={"earthquake": move})
        candidate = NS(species="celebi", active=False, fainted=False, moves={})
        opponent = NS(species="swampert", active=True, fainted=False, moves={"surf": incoming})
        return NS(active_pokemon=active, opponent_active_pokemon=opponent,
                  team={"active": active, "bench": candidate}, weather={})

    @staticmethod
    def _result(ko):
        return NS(reliable=True, ko_probability=1.0 if ko else 0.0)

    def test_non_guaranteed_incoming_damage_cannot_force_switch(self):
        battle = self._battle()
        with patch("battle_ai.override_gate.calculate_damage", return_value=self._result(False)):
            self.assertFalse(hard_loss_switch_allowed(battle, 0, 4))

    def test_guaranteed_model_ko_blocks_switch_override(self):
        battle = self._battle()
        calls = [self._result(True), self._result(True)]
        with patch("battle_ai.override_gate.calculate_damage", side_effect=calls):
            self.assertFalse(hard_loss_switch_allowed(battle, 0, 4))

    def test_guaranteed_incoming_ko_and_surviving_switch_allows_override(self):
        battle = self._battle()
        calls = [self._result(False), self._result(True), self._result(False)]
        with patch("battle_ai.override_gate.calculate_damage", side_effect=calls):
            self.assertTrue(hard_loss_switch_allowed(battle, 0, 4))

    def test_candidate_also_guaranteed_to_die_blocks_override(self):
        battle = self._battle()
        calls = [self._result(False), self._result(True), self._result(True)]
        with patch("battle_ai.override_gate.calculate_damage", side_effect=calls):
            self.assertFalse(hard_loss_switch_allowed(battle, 0, 4))


if __name__ == "__main__":
    unittest.main()
