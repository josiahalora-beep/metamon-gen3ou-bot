from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .damage import calculate_damage
from .state import snapshot_battle
from .insights import position_metadata
from .strategic import safety_override
from metamon.interface import consistent_move_order, consistent_pokemon_order


@dataclass(frozen=True)
class ActionEvaluation:
    action: int
    kind: str
    label: str
    tactical_score: float
    ko_probability: float
    reason: str


class TacticalEvaluator:
    """Conservative Gen 3 reranker with hard anti-throw safety checks."""

    def __init__(self, *, model_weight=1.0, damage_weight=0.35, ko_weight=2.0,
                 switch_penalty=0.15, anti_throw_penalty=2.0,
                 override_mode="verifier"):
        self.model_weight = model_weight
        self.damage_weight = damage_weight
        self.ko_weight = ko_weight
        self.switch_penalty = switch_penalty
        self.anti_throw_penalty = anti_throw_penalty
        self.override_mode = override_mode

    def evaluate(self, battle: Any, legal_actions: list[int], model_action: int) -> tuple[int, list[ActionEvaluation]]:
        state = snapshot_battle(battle, legal_actions)
        active = getattr(battle, "active_pokemon", None)
        target = getattr(battle, "opponent_active_pokemon", None)
        # Metamon's installed action contract is fixed-slot: 0..3 are the
        # alphabetically sorted active-move slots and 4..8 are sorted switches.
        raw_move_slots = list(getattr(active, "moves", {}).values())
        try:
            move_slots = consistent_move_order(raw_move_slots)
        except ValueError:
            move_slots = sorted(raw_move_slots, key=lambda m: str(getattr(m, "id", "")))
        available_move_ids = {getattr(m, "id", "") for m in (getattr(battle, "available_moves", []) or [])}
        raw_switch_slots = [p for p in getattr(battle, "team", {}).values() if not getattr(p, "fainted", False) and not getattr(p, "active", False)]
        try:
            switch_slots = consistent_pokemon_order(raw_switch_slots)
        except ValueError:
            switch_slots = sorted(raw_switch_slots, key=lambda p: str(getattr(p, "name", getattr(p, "species", ""))))
        evaluations = []
        for action in legal_actions:
            idx = int(action)
            is_move = 0 <= idx < 4 and not state.forced_switch
            if is_move:
                move = move_slots[idx] if idx < len(move_slots) else None
                if move is None or getattr(move, "id", "") not in available_move_ids:
                    evaluations.append(ActionEvaluation(idx, "illegal", f"move-slot:{idx}", -1e9, 0.0, "slot unavailable in current request"))
                    continue
                result = calculate_damage(active, target, move, weather=(state.weather[0] if state.weather else ""))
                evidence = result.reliable
                score = self.model_weight * float(idx == model_action)
                if evidence:
                    score += self.damage_weight * result.percentage_max + self.ko_weight * result.ko_probability
                label = str(getattr(move, "id", getattr(move, "name", "move")))
                reason = (f"damage {result.percentage_min:.1f}-{result.percentage_max:.1f}%; KO {result.ko_probability:.0%}"
                          if evidence else f"damage unavailable: {result.reason}")
                evaluations.append(ActionEvaluation(idx, "move", label, score, result.ko_probability, reason))
            else:
                switch_idx = idx - 4
                target_name = switch_slots[switch_idx].name if 0 <= switch_idx < len(switch_slots) else "unavailable"
                score = self.model_weight * float(idx == model_action) - self.switch_penalty
                evaluations.append(ActionEvaluation(idx, "switch", f"switch:{target_name}", score, 0.0, "switch position not numerically evaluated"))

        legal_set = {e.action for e in evaluations if e.kind != "illegal"}
        chosen = int(model_action) if int(model_action) in legal_set else (min(legal_set) if legal_set else 0)

        # Safety overrides are intentionally narrower than tactical reranking.
        # They are applied only for high-confidence anti-throw cases that use
        # revealed battle information and do not require invented opponent sets.
        if self.override_mode in {"verifier", "rerank"} and chosen in legal_set:
            decision = safety_override(battle, list(legal_set), chosen)
            if decision.action is not None and decision.action in legal_set and decision.action != chosen:
                chosen = decision.action
                for i, evaluation in enumerate(evaluations):
                    if evaluation.action == chosen:
                        evaluations[i] = ActionEvaluation(
                            evaluation.action,
                            evaluation.kind,
                            evaluation.label,
                            evaluation.tactical_score + self.anti_throw_penalty,
                            evaluation.ko_probability,
                            evaluation.reason + " | " + decision.reason,
                        )
                        break

        if self.override_mode == "rerank":
            best = max(evaluations, key=lambda x: x.tactical_score) if evaluations else None
            if best is not None and best.kind == "move" and best.tactical_score > -1e8:
                chosen = best.action
        return chosen, evaluations
