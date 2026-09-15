from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Reduce CPU oversubscription before importing the transformer stack.
_threads = max(1, min(8, os.cpu_count() or 4))
os.environ.setdefault("OMP_NUM_THREADS", str(_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(_threads))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

try:
    torch.set_num_threads(_threads)
    torch.set_num_interop_threads(max(1, min(4, _threads)))
except RuntimeError:
    # PyTorch may already have initialized its thread pools.
    pass

import play_public_synthetic as app


class CleanPublicQueueOnLadder(app.PublicQueueOnLadder):
    """Keep the existing transport but make traces reconstructable per turn."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_active_state = {}
        self._last_model = {}
        self._decision_started = {}

    @staticmethod
    def _action_labels(battle, action_count=9):
        moves = sorted(
            [str(getattr(m, "id", m)) for m in (getattr(battle, "available_moves", None) or [])],
            key=str.lower,
        )
        switches = sorted(
            [str(getattr(m, "species", m)) for m in (getattr(battle, "available_switches", None) or [])],
            key=str.lower,
        )
        labels = {}
        for i, move in enumerate(moves[:4]):
            labels[i] = f"move {move}"
        for i, mon in enumerate(switches[:5], start=4):
            labels[i] = f"switch {mon}"
        return labels, moves, switches

    def _trace(self, stage, battle_tag="", **fields):
        tag = str(battle_tag or "")

        if stage == "env.step.before" and tag:
            self._decision_started[tag] = datetime.now(timezone.utc)

        if stage == "action.model_selected" and tag:
            state = fields.get("state", {}) or {}
            moves = list(state.get("available_moves", []) or [])
            switches = list(state.get("available_switches", []) or [])
            canonical_moves = sorted(moves, key=str.lower)
            canonical_switches = sorted(switches, key=str.lower)
            model_action = int(fields.get("model_action", -1))
            labels = {}
            for i, move in enumerate(canonical_moves[:4]):
                labels[i] = f"move {move}"
            for i, mon in enumerate(canonical_switches[:5], start=4):
                labels[i] = f"switch {mon}"
            fields["model_action_label"] = labels.get(model_action, f"action {model_action}")
            fields["canonical_move_slots"] = canonical_moves[:4]
            fields["canonical_switch_slots"] = canonical_switches[:5]
            started = self._decision_started.get(tag)
            if started is not None:
                fields["decision_latency_ms"] = round(
                    (datetime.now(timezone.utc) - started).total_seconds() * 1000.0, 2
                )
            self._last_model[tag] = {
                "turn": int(state.get("turn", 0) or 0),
                "action": model_action,
                "label": fields["model_action_label"],
                "state": state,
            }

        elif stage == "action.final_selected" and tag:
            final_action = int(fields.get("final_action", -1))
            previous = self._last_model.get(tag, {})
            state = previous.get("state", {}) or {}
            canonical_moves = sorted(list(state.get("available_moves", []) or []), key=str.lower)
            canonical_switches = sorted(list(state.get("available_switches", []) or []), key=str.lower)
            labels = {}
            for i, move in enumerate(canonical_moves[:4]):
                labels[i] = f"move {move}"
            for i, mon in enumerate(canonical_switches[:5], start=4):
                labels[i] = f"switch {mon}"
            fields["final_action_label"] = labels.get(final_action, f"action {final_action}")
            if previous:
                fields["model_action_label"] = previous.get("label")
                fields["decision_latency_ms"] = previous.get("decision_latency_ms")

        elif stage == "action.converted_to_order" and tag:
            # Emit one compact, analysis-friendly record for each actual decision.
            model = self._last_model.get(tag, {})
            final_action = int(fields.get("final_action", -1))
            state = fields.get("state", {}) or model.get("state", {}) or {}
            canonical_moves = sorted(list(state.get("available_moves", []) or []), key=str.lower)
            canonical_switches = sorted(list(state.get("available_switches", []) or []), key=str.lower)
            labels = {}
            for i, move in enumerate(canonical_moves[:4]):
                labels[i] = f"move {move}"
            for i, mon in enumerate(canonical_switches[:5], start=4):
                labels[i] = f"switch {mon}"
            self._base_trace(
                "decision",
                tag,
                turn=int(state.get("turn", 0) or 0),
                active=state.get("active", ""),
                opponent_active=state.get("opponent_active", ""),
                model_action=model.get("action"),
                model_action_label=model.get("label"),
                final_action=final_action,
                final_action_label=labels.get(final_action, f"action {final_action}"),
                override=bool(model.get("action") is not None and model.get("action") != final_action),
                command=fields.get("command", ""),
                decision_latency_ms=self._decision_started.pop(tag, None),
            )
            self._last_active_state.pop(tag, None)

        # Suppress repeated identical lifecycle heartbeats. Stall snapshots remain.
        if stage == "lifecycle.battle_active" and tag:
            state = fields.get("state")
            if self._last_active_state.get(tag) == state:
                return
            self._last_active_state[tag] = state

        super()._trace(stage, tag, **fields)

    def _base_trace(self, stage, battle_tag="", **fields):
        if self.transport_trace is not None:
            self.transport_trace.event(stage, battle_tag, **fields)


# Keep the existing factory/runner code unchanged; only replace the environment class
# that the factory resolves at runtime.
app.PublicQueueOnLadder = CleanPublicQueueOnLadder


def _is_transport_trace_arg(value: str) -> bool:
    return value in {"--transport-trace"}


def _make_unique_trace_path(argv: list[str]) -> list[str]:
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_path = Path("battle_data") / "ladder20" / f"transport_trace_{run_stamp}.jsonl"
    output = list(argv)
    for i, value in enumerate(output):
        if _is_transport_trace_arg(value) and i + 1 < len(output):
            output[i + 1] = str(default_path)
            return output
    output.extend(["--transport-trace", str(default_path)])
    return output


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *_make_unique_trace_path(sys.argv[1:])]
    try:
        app.main = app.main
        import asyncio
        asyncio.run(app.main())
    except KeyboardInterrupt:
        print("\n[runner] stopped by Ctrl+C; partial battle/decision data was preserved.", flush=True)
        raise SystemExit(130)
