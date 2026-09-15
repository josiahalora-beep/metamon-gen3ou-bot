import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from battle_ai.smogon_priors import expected_stats, load_profiles


class SmogonPriorTests(unittest.TestCase):
    def test_revealed_move_reorders_compatible_sets(self):
        payload = {
            "Starmie": {
                "Standard": {
                    "moves": ["Surf", "Psychic", "Recover", "Rapid Spin"],
                    "item": "Leftovers",
                    "nature": "Timid",
                    "evs": {"spa": 252, "spe": 252},
                },
                "Thunder": {
                    "moves": ["Thunder", "Surf", "Ice Beam", "Recover"],
                    "item": "Leftovers",
                    "nature": "Timid",
                    "evs": {"spa": 252, "spe": 252},
                },
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gen3ou.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            profiles = load_profiles("starmie", revealed_moves=("thunder",), path=path)
        self.assertEqual(profiles[0].name, "Thunder")
        self.assertIn("thunder", profiles[0].evidence)
        self.assertGreater(profiles[0].weight, profiles[1].weight)

    def test_31_iv_competitive_stat_reconstruction(self):
        payload = {
            "Blissey": {
                "Special Wall": {
                    "moves": ["Soft-Boiled", "Seismic Toss", "Thunder Wave", "Ice Beam"],
                    "item": "Leftovers",
                    "nature": "Calm",
                    "evs": {"hp": 252, "spd": 252},
                }
            }
        }
        pokemon = SimpleNamespace(
            species="Blissey",
            base_stats={"hp": 255, "atk": 10, "def": 10, "spa": 75, "spd": 135, "spe": 55},
            level=100,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gen3ou.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            profile = load_profiles("Blissey", path=path)[0]
            stats = expected_stats(pokemon, profile)
        self.assertGreater(stats["spd"], 500)
        self.assertGreater(stats["hp"], 600)


if __name__ == "__main__":
    unittest.main()
