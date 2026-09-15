from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .damage import calculate_damage
from .state import snapshot_battle
from .strategic import safety_override
from .strategic_plan import strategic_opportunity_override
from .threat_response import hidden_threat_switch_override
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
    """Conservative Gen 3 verifier/reranker with strategic anti-throw checks."""

    def __init__(self, *, model_weight=1.0, damage_weight=0.35, ko_weight=2.0,
                 switch_penalty=0.15, anti_throw_penalty=2.0,
                 override_mode="verifier"):
        self.model_weight = model_weight
        self.damage_weight = damage_weight
        self.ko_weight = ko_weight
        self.switch_penalty = switch_penalty
        self.anti_throw_penalty = anti_throw_penalty
        self.override_mode = override_mode

    @staticmethod
    def _is_passive_move(move: Any) -> bool:
        """Moves that do not directly deal damage this turn."""
        return int(getattr(move, "base_power", 0) or 0) <= 0

    @staticmethod
    def _is_protect_counter_move(move: Any) -> bool:
        """Protect/Detect/Endure-style moves sharing Gen 3's streak counter."""
        return bool(getattr(move, "is_protect_counter", False))

    @staticmethod
    def _protect_counter(active: Any) -> int:
        """Consecutive successful Protect/Detect/Endure uses tracked by poke-env."""
        try:
            return max(0, int(getattr(active, "_protect_counter", 0) or 0))
        except (TypeError, ValueError):
            return 0

    def _protect_sequence_breaker(self, active: Any, target: Any, move_slots: list[Any],
                                  evaluations: list[ActionEvaluation], model_action: int,
                                  weather: str = "") -> tuple[int | None, str]:
        """Avoid knowingly spending another Gen 3 protection-streak roll."""
        if not (0 <= model_action < 4) or model_action >= len(move_slots):
            return None, ""
        selected = move_slots[model_action]
        if selected is None or not self._is_protect_counter_move(selected):
            return None, ""
        counter = self._protect_counter(active)
        if counter <= 0:
            return None, ""

        candidates = []
        for evaluation in evaluations:
            if evaluation.kind != "move" or evaluation.action == model_action:
                continue
            if not (0 <= evaluation.action < len(move_slots)):
                continue
            move = move_slots[evaluation.action]
            if move is None or self._is_passive_move(move):
                continue
            result = calculate_damage(active, target, move, weather=weather)
            if not result.reliable:
                continue
            candidates.append((result.ko_probability, result.max_damage, -evaluation.action, evaluation.action, move))
        if not candidates:
            return None, ""

        _, _, _, action, move = max(candidates)
        move_name = getattr(move, "name", getattr(move, "id", "attack"))
        success_probability = 1.0 / (2 ** counter)
        return action, (
            f"protect sequence breaker: {getattr(selected, 'name', getattr(selected, 'id', 'protect'))} "
            f"has already succeeded {counter} consecutive time(s); next success is only "
            f"{success_probability:.1%}; use {move_name} instead"
        )

    def evaluate(self, battle: Any, legal_actions: list[int], model_action: int) -> tuple[int, list[ActionEvaluation]]:
        state = snapshot_battle(battle, legal_actions)
        active = getattr(battle, "active_pokemon", None)
        target = getattr(battle, "opponent_active_pokemon", None)
        raw_move_slots = list(getattr(active, "moves", {}).values())
        try:
            move_slots = consistent_move_order(raw_move_slots)
        except ValueError:
            move_slots = sorted(raw_move_slots, key=lambda m: str(getattr(m, "id", "")))
        available_move_ids = {getattr(m, "id", "") for m in (getattr(battle, "available_moves", []) or [])}
        raw_switch_slots = [p for p in getattr(battle, "team", {}).values()
                            if not getattr(p, "fainted", False) and not getattr(p, "active", False)]
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
                score = self.model_weight * float(idx == model_action)
                if result.reliable:
                    score += self.damage_weight * result.percentage_max + self.ko_weight * result.ko_probability
                label = str(getattr(move, "id", getattr(move, "name", "move")))
                reason = (f"damage {result.percentage_min:.1f}-{result.percentage_max:.1f}%; KO {result.ko_probability:.0%}"
                          if result.reliable else f"damage unavailable: {result.reason}")
                if result.reason:
                    reason += f" [{result.reason}]"
                evaluations.append(ActionEvaluation(idx, "move", label, score, result.ko_probability, reason))
            else:
                switch_idx = idx - 4
                target_name = switch_slots[switch_idx].name if 0 <= switch_idx < len(switch_slots) else "unavailable"
                score = self.model_weight * float(idx == model_action) - self.switch_penalty
                evaluations.append(ActionEvaluation(idx, "switch", f"switch:{target_name}", score, 0.0, "switch position not numerically evaluated"))

        legal_set = {e.action for e in evaluations if e.kind != "illegal"}
        chosen = int(model_action) if int(model_action) in legal_set else (min(legal_set) if legal_set else 0)
        safety_action = None

        sequence_action, sequence_reason = self._protect_sequence_breaker(
            active, target, move_slots, evaluations, chosen,
            weather=(state.weather[0] if state.weather else ""),
        )
        if sequence_action is not None and sequence_action in legal_set and sequence_action != chosen:
            chosen = sequence_action
            safety_action = sequence_action
            for i, evaluation in enumerate(evaluations):
                if evaluation.action == chosen:
                    evaluations[i] = ActionEvaluation(
                        evaluation.action, evaluation.kind, evaluation.label,
                        evaluation.tactical_score + self.anti_throw_penalty,
                        evaluation.ko_probability,
                        evaluation.reason + " | " + sequence_reason,
                    )
                    break

        # Hidden-stat battles can reach positions where the opponent's moves are
        # not revealed yet. Give the model one last chance to attack/act, but do
        # not permit a passive turn when broad speed/offense evidence points to a
        # stronger opposing attacker and a clearly better defensive teammate.
        if safety_action is None:
            threat_action, threat_reason = hidden_threat_switch_override(
                battle, list(legal_set), chosen,
                {idx: move for idx, move in enumerate(move_slots[:4]) if move is not None},
            )
            if threat_action is not None and threat_action in legal_set and threat_action != chosen:
                chosen = threat_action
                safety_action = threat_action
                for i, evaluation in enumerate(evaluations):
                    if evaluation.action == chosen:
                        evaluations[i] = ActionEvaluation(
                            evaluation.action, evaluation.kind, evaluation.label,
                            evaluation.tactical_score + self.anti_throw_penalty,
                            evaluation.ko_probability,
                            evaluation.reason + " | " + threat_reason,
                        )
                        break

        if safety_action is None:
            strategic_action, strategic_reason = strategic_opportunity_override(
                battle, list(legal_set), chosen
            )
            if strategic_action is not None and strategic_action in legal_set and strategic_action != chosen:
                chosen = strategic_action
                safety_action = strategic_action
                for i, evaluation in enumerate(evaluations):
                    if evaluation.action == chosen:
                        evaluations[i] = ActionEvaluation(
                            evaluation.action, evaluation.kind, evaluation.label,
                            evaluation.tactical_score + self.anti_throw_penalty,
                            evaluation.ko_probability,
                            evaluation.reason + " | " + strategic_reason,
                        )
                        break

        if self.override_mode in {"verifier", "rerank"} and chosen in legal_set and safety_action is None:
            decision = safety_override(battle, list(legal_set), chosen)
            if decision.action is not None and decision.action in legal_set and decision.action != chosen:
                chosen = decision.action
                safety_action = decision.action
                for i, evaluation in enumerate(evaluations):
                    if evaluation.action == chosen:
                        evaluations[i] = ActionEvaluation(
                            evaluation.action, evaluation.kind, evaluation.label,
                            evaluation.tactical_score + self.anti_throw_penalty,
                            evaluation.ko_probability,
                            evaluation.reason + " | " + decision.reason,
                        )
                        break

        if self.override_mode == "rerank" and safety_action is None:
            best = max(evaluations, key=lambda x: x.tactical_score) if evaluations else None
            if best is not None and best.kind == "move" and best.tactical_score > -1e8:
                chosen = best.action
        return chosen, evaluations
