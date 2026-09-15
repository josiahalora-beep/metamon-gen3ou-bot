from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from battle_ai.damage import DamageRange
from battle_ai.intelligence_guard import immediate_loss_guard


def pokemon(name: str, *, hp: float = 1.0, moves=(), active=False, fainted=False):
    return SimpleNamespace(
        name=name,
        species=name,
        current_hp_fraction=hp,
        fainted=fainted,
        active=active,
        moves={f"{i}": move for i, move in enumerate(moves)},
        types=(),
        boosts={},
        level=100,
        base_stats={"hp": 100, "atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100},
        max_hp=400,
        current_hp=int(400 * hp),
        status="",
    )


def move(move_id: str, power: int):
    return SimpleNamespace(id=move_id, name=move_id, base_power=power, type="normal")


def battle(active, opponent, switches):
    return SimpleNamespace(
        active_pokemon=active,
        opponent_active_pokemon=opponent,
        team={p.name: p for p in [active, *switches]},
        weather={},
    )


def dmg(minimum=10, maximum=20, ko=0.0):
    return DamageRange(minimum, maximum, minimum / 4, maximum / 4, ko, reliable=True, reason="")


def test_rejects_nonwinning_self_ko_for_safe_switch():
    explosion = move("explosion", 250)
    active = pokemon("gengar", moves=(explosion,), active=True)
    opponent = pokemon("celebi", moves=())
    switch = pokemon("blissey", hp=1.0)
    state = battle(active, opponent, [switch])

    with patch("battle_ai.intelligence_guard.calculate_damage", return_value=dmg(30, 40, 0.0)):
        action, reason = immediate_loss_guard(state, [0, 4], 0)

    assert action == 4
    assert "self-KO" in reason


def test_rejects_attack_into_revealed_guaranteed_ko():
    attack = move("thunderbolt", 95)
    incoming = move("earthquake", 100)
    active = pokemon("gengar", hp=0.50, moves=(attack,), active=True)
    opponent = pokemon("metagross", moves=(incoming,))
    switch = pokemon("skarmory", hp=1.0)
    state = battle(active, opponent, [switch])

    def calc(attacker, defender, selected, **_):
        if attacker is opponent:
            return dmg(300, 400, 1.0)
        return dmg(20, 40, 0.0)

    with patch("battle_ai.intelligence_guard.calculate_damage", side_effect=calc):
        action, reason = immediate_loss_guard(state, [0, 4], 0)

    assert action == 4
    assert "guaranteed KO" in reason


def test_keeps_guaranteed_winning_attack():
    attack = move("icebeam", 95)
    incoming = move("earthquake", 100)
    active = pokemon("starmie", hp=0.50, moves=(attack,), active=True)
    opponent = pokemon("salamence", moves=(incoming,))
    switch = pokemon("blissey", hp=1.0)
    state = battle(active, opponent, [switch])

    def calc(attacker, defender, selected, **_):
        if attacker is active:
            return dmg(300, 400, 1.0)
        return dmg(300, 400, 1.0)

    with patch("battle_ai.intelligence_guard.calculate_damage", side_effect=calc):
        action, _ = immediate_loss_guard(state, [0, 4], 0)

    assert action is None


def test_rejects_switch_that_is_guaranteed_to_die():
    switch_bad = pokemon("blissey", hp=1.0)
    switch_good = pokemon("skarmory", hp=1.0)
    active = pokemon("swampert", hp=1.0, active=True)
    incoming = move("earthquake", 100)
    opponent = pokemon("tyranitar", moves=(incoming,))
    state = battle(active, opponent, [switch_bad, switch_good])

    def calc(attacker, defender, selected, **_):
        if defender is switch_bad:
            return dmg(300, 400, 1.0)
        return dmg(20, 40, 0.0)

    with patch("battle_ai.intelligence_guard.calculate_damage", side_effect=calc):
        action, reason = immediate_loss_guard(state, [4, 5], 4)

    assert action == 5
    assert "guaranteed to be KO'd" in reason
