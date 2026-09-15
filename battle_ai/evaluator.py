from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .damage import calculate_damage
from .state import snapshot_battle
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
    """Conservative Gen 3 verifier/reranker with hard anti-throw checks."""

    def __init__(self, *, model_weight=1.0, damage_weight=0.35, ko_weight=2.0,
                 switch_penalty=0.15, anti_throw_penalty=2.0,
                 override_mode="verifier"):
        self.model_weight = model_weight
        self.damage_weight = damage_weight
        self.ko_weight = ko_weight
        self.switch_penalty = switch_penalty
        self.anti_throw_penalty = anti_throw_penalty
        self.override_mode = override_mode
        self._history_battle_tag = None
        self._consecutive_passive_action = None
        self._consecutive_passive_count = 0

    @staticmethod
    def _is_passive_move(move: Any) -> bool:
        """Moves that do not directly pressure the opponent this turn."""
        return int(getattr(move, "base_power", 0) or 0) <= 0

    def _note_model_action(self, battle: Any, model_action: int, move_slots: list[Any]) -> None:
        """Track repeated passive choices for one battle without leaking across episodes."""
        tag = str(getattr(battle, "battle_tag", ""))
        if tag != self._history_battle_tag:
            self._history_battle_tag = tag
            self._consecutive_passive_action = None
            self._consecutive_passive_count = 0
        move = move_slots[model_action] if 0 <= model_action < len(move_slots) else None
        if move is not None and self._is_passive_move(move):
            if self._consecutive_passive_action == model_action:
                self._consecutive_passive_count += 1
            else:
                self._consecutive_passive_action = model_action
                self._consecutive_passive_count = 1
        else:
            self._consecutive_passive_action = None
            self._consecutive_passive_count = 0

    def _passive_loop_breaker(self, active: Any, target: Any, legal_actions: list[int],
                              move_slots: list[Any], evaluations: list[ActionEvaluation],
                              model_action: int) -> tuple[int | None, str]:
        """Break repeated Protect/setup/stall loops before the Pokemon is slowly lost."""
        if self._consecutive_passive_count < 2 or not (0 <= model_action < 4):
            return None, ""
        selected = move_slots[model_action] if model_action < len(move_slots) else None
        if selected is None or not self._is_passive_move(selected):
            return None, ""

        candidates = []
        for evaluation in evaluations:
            if evaluation.kind != "move" or evaluation.action == model_action:
                continue
            move = move_slots[evaluation.action] if evaluation.action < len(move_slots) else None
            if move is None or self._is_passive_move(move):
                continue
            result = calculate_damage(active, target, move, weather="")
            if not result.reliable:
                continue
            candidates.append((result.ko_probability, result.max_damage, -evaluation.action, evaluation.action, move))

        if not candidates:
            return None, ""
        _, _, _, action, move = max(candidates)
        move_name = getattr(move, "name", getattr(move, "id", "attack"))
        return action, (
            f"passive-loop breaker: {getattr(selected, 'name', getattr(selected, 'id', 'passive'))} "
            f"was selected {self._consecutive_passive_count} consecutive times; use {move_name} instead"
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
        self._note_model_action(battle, int(model_action), move_slots)
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

        loop_action, loop_reason = self._passive_loop_breaker(
            active, target, legal_actions, move_slots, evaluations, chosen
        )
        if loop_action is not None and loop_action in legal_set:
            chosen = loop_action
            safety_action = loop_action
            for i, evaluation in enumerate(evaluations):
                if evaluation.action == chosen:
                    evaluations[i] = ActionEvaluation(
                        evaluation.action, evaluation.kind, evaluation.label,
                        evaluation.tactical_score + self.anti_throw_penalty,
                        evaluation.ko_probability,
                        evaluation.reason + " | " + loop_reason,
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

        # Reranking may optimize among ordinary actions, but it must never
        # undo a hard anti-throw correction. This makes the safety contract
        # invariant across verifier and rerank configurations.
        if self.override_mode == "rerank" and safety_action is None:
            best = max(evaluations, key=lambda x: x.tactical_score) if evaluations else None
            if best is not None and best.kind == "move" and best.tactical_score > -1e8:
                chosen = best.action
        return chosen, evaluations
