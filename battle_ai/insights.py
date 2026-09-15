from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict


@dataclass(frozen=True)
class SetHypothesis:
    name: str
    confidence: float
    evidence: tuple[str, ...]


def infer_opponent_set(pokemon: dict) -> list[SetHypothesis]:
    """Score broad ADV roles without pretending an unrevealed set is known."""
    moves = {m.get("name", "") for m in pokemon.get("moves", ())}
    out = []
    if any(x in moves for x in {"agility", "rockpolish"}):
        out.append(SetHypothesis("setup_offense", 0.85, tuple(sorted(moves & {"agility", "rockpolish"}))))
    if any(x in moves for x in {"spikes", "rapidspin", "stealthrock"}):
        out.append(SetHypothesis("hazard_control", 0.8, tuple(sorted(moves & {"spikes", "rapidspin", "stealthrock"}))))
    if any(x in moves for x in {"rest", "protect", "toxic", "roar", "whirlwind"}):
        out.append(SetHypothesis("stability_or_stall", 0.55, tuple(sorted(moves & {"rest", "protect", "toxic", "roar", "whirlwind"}))))
    if not out:
        out.append(SetHypothesis("unknown", 0.2, tuple(sorted(moves))))
    return out


def threat_score(pokemon: dict, our_team: tuple[dict, ...]) -> dict:
    hp = float(pokemon.get("hp_fraction", 0.0))
    boosts = pokemon.get("boosts", {})
    setup = max([int(v) for v in boosts.values()] or [0])
    hypotheses = infer_opponent_set(pokemon)
    score = min(10, 2 + int(4 * hp) + 2 * max(0, setup) + (2 if hypotheses[0].name == "setup_offense" else 0))
    return {"threat_level": score, "likely_role": hypotheses[0].name, "checks_remaining": sum(1 for p in our_team if not p.get("fainted")), "hypotheses": [h.__dict__ for h in hypotheses]}


def position_metadata(state) -> dict:
    opp = state.opponent_active
    return {
        "opponent_set_inference": [h.__dict__ for p in state.opponent_team for h in infer_opponent_set(p)],
        "active_threat": threat_score(opp, state.our_team) if opp else {},
        "win_condition": {"our_remaining": sum(not p.get("fainted") for p in state.our_team), "opponent_remaining": sum(not p.get("fainted") for p in state.opponent_team)},
        "hazards": {"ours": state.our_hazards, "opponent": state.opponent_hazards},
        "weather": list(state.weather),
        "status": {"ours": state.our_active.get("status", ""), "opponent": opp.get("status", "") if opp else ""},
    }
