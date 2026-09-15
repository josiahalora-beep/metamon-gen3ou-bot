from types import SimpleNamespace

from battle_ai import intelligence_guard_v2 as guard


def test_demonstrated_move_blocks_nonwinning_attack(monkeypatch):
    earthquake = SimpleNamespace(
        id="earthquake",
        name="Earthquake",
        base_power=100,
        current_pp=9,
        max_pp=10,
    )
    weak_attack = SimpleNamespace(
        id="icebeam",
        name="Ice Beam",
        base_power=95,
        current_pp=9,
        max_pp=10,
    )

    active = SimpleNamespace(
        current_hp=144,
        current_hp_fraction=144 / 387,
        moves={"icebeam": weak_attack},
        species="Tyranitar",
        name="Tyranitar",
        active=True,
        fainted=False,
    )
    opponent = SimpleNamespace(
        moves={"earthquake": earthquake},
        species="Gyarados",
        name="Gyarados",
        active=True,
        fainted=False,
    )
    safe_switch = SimpleNamespace(
        name="Suicune",
        species="Suicune",
        current_hp_fraction=1.0,
        active=False,
        fainted=False,
    )
    unsafe_switch = SimpleNamespace(
        name="Dugtrio",
        species="Dugtrio",
        current_hp_fraction=1.0,
        active=False,
        fainted=False,
    )
    battle = SimpleNamespace(
        active_pokemon=active,
        opponent_active_pokemon=opponent,
        team={"a": safe_switch, "b": unsafe_switch},
        weather={},
    )

    def fake_damage(attacker, defender, move, **kwargs):
        if attacker is opponent and defender is active and move is earthquake:
            return SimpleNamespace(
                reliable=True,
                ko_probability=0.0,
                max_damage=243,
                percentage_max=168.75,
                percentage_min=120.0,
                reason="estimated",
            )
        return SimpleNamespace(
            reliable=True,
            ko_probability=0.0,
            max_damage=50,
            percentage_max=35.0,
            percentage_min=20.0,
            reason="",
        )

    monkeypatch.setattr(guard, "calculate_damage", fake_damage)
    monkeypatch.setattr(guard, "_safe_switch_action", lambda battle, legal: 5)

    action, reason = guard.immediate_loss_guard(battle, [0, 4, 5], 0)

    assert action == 5
    assert "demonstrated earthquake" in reason
