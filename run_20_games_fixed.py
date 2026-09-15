from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_threads = max(1, min(8, os.cpu_count() or 4))
os.environ.setdefault("OMP_NUM_THREADS", str(_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(_threads))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

try:
    torch.set_num_threads(_threads)
    torch.set_num_interop_threads(max(1, min(4, _threads)))
except RuntimeError:
    pass

import play_public_synthetic as app


class CleanPublicQueueOnLadder(app.PublicQueueOnLadder):
    """Keep existing ladder behavior while making every decision traceable."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_active_state = {}
        self._last_model = {}
        self._decision_started = {}

    def _labels_for_state(self, state):
        moves = sorted(list(state.get("available_moves", []) or []), key=str.lower)
        switches = sorted(list(state.get("available_switches", []) or []), key=str.lower)
        labels = {}
        for i, move in enumerate(moves[:4]):
            labels[i] = f"move {move}"
        for i, mon in enumerate(switches[:5], start=4):
            labels[i] = f"switch {mon}"
        return labels, moves[:4], switches[:5]

    def _trace(self, stage, battle_tag="", **fields):
        tag = str(battle_tag or "")

        if stage == "env.step.before" and tag:
            self._decision_started[tag] = datetime.now(timezone.utc)

        if stage == "action.model_selected" and tag:
            state = fields.get("state", {}) or {}
            labels, moves, switches = self._labels_for_state(state)
            model_action = int(fields.get("model_action", -1))
            fields["model_action_label"] = labels.get(model_action, f"action {model_action}")
            fields["canonical_move_slots"] = moves
            fields["canonical_switch_slots"] = switches
            started = self._decision_started.get(tag)
            latency_ms = None
            if started is not None:
                latency_ms = round(
                    (datetime.now(timezone.utc) - started).total_seconds() * 1000.0,
                    2,
                )
                fields["decision_latency_ms"] = latency_ms
            self._last_model[tag] = {
                "turn": int(state.get("turn", 0) or 0),
                "action": model_action,
                "label": fields["model_action_label"],
                "state": state,
                "latency_ms": latency_ms,
            }

        elif stage == "action.final_selected" and tag:
            final_action = int(fields.get("final_action", -1))
            previous = self._last_model.get(tag, {})
            state = previous.get("state", {}) or {}
            labels, _, _ = self._labels_for_state(state)
            fields["final_action_label"] = labels.get(final_action, f"action {final_action}")
            if previous:
                fields["model_action_label"] = previous.get("label")
                fields["decision_latency_ms"] = previous.get("latency_ms")

        elif stage == "action.converted_to_order" and tag:
            model = self._last_model.get(tag, {})
            state = fields.get("state", {}) or model.get("state", {}) or {}
            labels, moves, switches = self._labels_for_state(state)
            final_action = int(fields.get("final_action", -1))
            model_action = model.get("action")
            self._base_trace(
                "decision",
                tag,
                turn=int(state.get("turn", 0) or 0),
                active=state.get("active", ""),
                opponent_active=state.get("opponent_active", ""),
                model_action=model_action,
                model_action_label=model.get("label"),
                final_action=final_action,
                final_action_label=labels.get(final_action, f"action {final_action}"),
                override=bool(model_action is not None and model_action != final_action),
                command=fields.get("command", ""),
                canonical_move_slots=moves,
                canonical_switch_slots=switches,
                decision_latency_ms=model.get("latency_ms"),
            )
            self._decision_started.pop(tag, None)

        # lifecycle.battle_active currently fires once per second in the ladder loop.
        # Keep the first record and only write later records when the compact state changes.
        if stage == "lifecycle.battle_active" and tag:
            state = fields.get("state")
            if self._last_active_state.get(tag) == state:
                return
            self._last_active_state[tag] = state

        super()._trace(stage, tag, **fields)

    def _base_trace(self, stage, battle_tag="", **fields):
        if self.transport_trace is not None:
            self.transport_trace.event(stage, battle_tag, **fields)


app.PublicQueueOnLadder = CleanPublicQueueOnLadder


def _make_unique_trace_path(argv: list[str]) -> list[str]:
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_path = Path("battle_data") / "ladder20" / f"transport_trace_{run_stamp}.jsonl"
    output = list(argv)
    for index, value in enumerate(output):
        if value == "--transport-trace" and index + 1 < len(output):
            output[index + 1] = str(trace_path)
            return output
    output.extend(["--transport-trace", str(trace_path)])
    return output


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *_make_unique_trace_path(sys.argv[1:])]
    try:
        import asyncio
        asyncio.run(app.main())
    except KeyboardInterrupt:
        print("\n[runner] stopped by Ctrl+C; partial battle/decision data was preserved.", flush=True)
        raise SystemExit(130)
