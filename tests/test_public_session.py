import unittest
import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from play_public_synthetic import (
    BattleSessionState,
    PublicQueueOnLadder,
    SessionPhase,
    TransportTraceWriter,
)


class MemoryTrace:
    def __init__(self):
        self.events = []

    def event(self, stage, battle_tag="", **fields):
        self.events.append({"stage": stage, "battle_tag": battle_tag, **fields})


class PublicSessionLifecycleTests(unittest.TestCase):
    def test_sequential_battle_lifecycle_clears_tag(self):
        session = BattleSessionState()
        session.begin_queue()
        session.battle_created("battle-1")
        session.battle_started("battle-1")
        session.turn_received("battle-1")
        session.battle_ended("battle-1")
        session.cleanup_started()
        session.cleanup_finished()
        self.assertEqual(session.phase, SessionPhase.IDLE)
        self.assertIsNone(session.battle_tag)

        session.begin_queue()
        session.battle_created("battle-2")
        session.battle_started("battle-2")
        self.assertEqual(session.battle_tag, "battle-2")
        self.assertNotEqual(session.generation, 0)

    def test_stale_battle_cannot_replace_active_battle(self):
        session = BattleSessionState()
        session.begin_queue()
        session.battle_created("battle-1")
        with self.assertRaises(RuntimeError):
            session.battle_created("battle-2")

    def test_stale_turn_is_rejected(self):
        session = BattleSessionState()
        session.begin_queue()
        session.battle_created("battle-1")
        session.battle_started("battle-1")
        with self.assertRaises(RuntimeError):
            session.turn_received("battle-2")

    def test_completed_battle_cannot_receive_action(self):
        wrapper = PublicQueueOnLadder.__new__(PublicQueueOnLadder)
        wrapper.current_battle = SimpleNamespace(battle_tag="battle-1", finished=True)
        wrapper._session = BattleSessionState(SessionPhase.ENDED, "battle-1", 1)
        with self.assertRaises(RuntimeError):
            wrapper.action_to_move(0, wrapper.current_battle)

    def test_old_object_cannot_control_new_battle(self):
        wrapper = PublicQueueOnLadder.__new__(PublicQueueOnLadder)
        old = SimpleNamespace(battle_tag="battle-1", finished=False)
        new = SimpleNamespace(battle_tag="battle-2", finished=False)
        wrapper.current_battle = new
        wrapper._session = BattleSessionState(SessionPhase.ACTIVE, "battle-2", 2)
        with self.assertRaises(RuntimeError):
            wrapper.action_to_move(0, old)

    def test_queue_cancellation_cleans_without_sleep(self):
        session = BattleSessionState()
        session.begin_queue()
        session.queue_cancelled()
        session.cleanup_finished()
        self.assertEqual(session.phase, SessionPhase.IDLE)
        self.assertIsNone(session.battle_tag)

    def test_transport_debug_redacts_credentials_and_teams(self):
        self.assertEqual(
            TransportTraceWriter.redact_text("/trn JorelMorell,0,secret-assertion"),
            "/trn JorelMorell,0,[redacted-assertion]",
        )
        self.assertEqual(
            TransportTraceWriter.redact_text("/utm packed-team"),
            "/utm [redacted-team]",
        )
        self.assertEqual(
            TransportTraceWriter.redact_text("battle-room|/utm packed-team"),
            "battle-room|/utm [redacted-team]",
        )

    def test_transport_hook_records_choose_before_and_after_websocket_send(self):
        async def run_probe():
            trace = MemoryTrace()
            sent = []

            async def send_message(message, room="", message_2=None):
                sent.append((room, message, message_2))

            async def handle_message(message):
                return None

            wrapper = PublicQueueOnLadder.__new__(PublicQueueOnLadder)
            wrapper.transport_trace = trace
            wrapper._last_transport_progress = datetime.now(timezone.utc)
            wrapper._stall_reported_for = None
            wrapper.agent = SimpleNamespace(
                ps_client=SimpleNamespace(
                    send_message=send_message,
                    _handle_message=handle_message,
                )
            )
            wrapper._install_transport_trace_hooks()
            await wrapper.agent.ps_client.send_message("/choose move earthquake", "battle-gen3ou-test")
            return trace.events, sent

        events, sent = asyncio.run(run_probe())
        self.assertEqual(sent, [("battle-gen3ou-test", "/choose move earthquake", None)])
        stages = [event["stage"] for event in events]
        self.assertIn("websocket.outbound.before_send", stages)
        self.assertIn("websocket.outbound.after_send", stages)
        outbound = [event for event in events if event["stage"] == "websocket.outbound.after_send"][0]
        self.assertEqual(outbound["message"], "/choose move earthquake")
        self.assertEqual(outbound["battle_tag"], "battle-gen3ou-test")

    def test_transport_hook_records_raw_inbound_room_traffic(self):
        async def run_probe():
            trace = MemoryTrace()

            async def send_message(message, room="", message_2=None):
                return None

            async def handle_message(message):
                return "handled"

            wrapper = PublicQueueOnLadder.__new__(PublicQueueOnLadder)
            wrapper.transport_trace = trace
            wrapper._last_transport_progress = datetime.now(timezone.utc)
            wrapper._stall_reported_for = None
            wrapper.agent = SimpleNamespace(
                ps_client=SimpleNamespace(
                    send_message=send_message,
                    _handle_message=handle_message,
                )
            )
            wrapper._install_transport_trace_hooks()
            result = await wrapper.agent.ps_client._handle_message(">battle-gen3ou-test\n|turn|2")
            return trace.events, result

        events, result = asyncio.run(run_probe())
        self.assertEqual(result, "handled")
        inbound = [event for event in events if event["stage"] == "websocket.inbound.raw"][0]
        self.assertEqual(inbound["battle_tag"], "battle-gen3ou-test")
        self.assertIn("|turn|2", inbound["message"])

    def test_stall_snapshot_records_current_state_without_sending_action(self):
        async def run_probe():
            trace = MemoryTrace()
            wrapper = PublicQueueOnLadder.__new__(PublicQueueOnLadder)
            wrapper.transport_trace = trace
            wrapper.transport_stall_seconds = 0
            wrapper._last_transport_progress = datetime.now(timezone.utc)
            wrapper._stall_reported_for = None
            wrapper._session = BattleSessionState(SessionPhase.ACTIVE, "battle-gen3ou-test", 1)
            battle = SimpleNamespace(
                battle_tag="battle-gen3ou-test",
                turn=1,
                finished=False,
                won=False,
                lost=False,
                _wait=False,
                trapped=False,
                active_pokemon=SimpleNamespace(species="Aerodactyl"),
                opponent_active_pokemon=SimpleNamespace(species="Tyranitar"),
                available_moves=[SimpleNamespace(id="earthquake")],
                available_switches=[],
            )
            wrapper.agent = SimpleNamespace(
                current_battle=battle,
                ps_client=SimpleNamespace(websocket=SimpleNamespace(closed=False)),
                actions=SimpleNamespace(empty=lambda: True),
                observations=SimpleNamespace(empty=lambda: True),
                _battles={"battle-gen3ou-test": battle},
            )
            wrapper._maybe_trace_stall(SimpleNamespace(done=lambda: False))
            return trace.events

        events = asyncio.run(run_probe())
        snapshots = [event for event in events if event["stage"] == "transport.stall_snapshot"]
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0]["battle_tag"], "battle-gen3ou-test")
        self.assertEqual(snapshots[0]["state"]["turn"], 1)


if __name__ == "__main__":
    unittest.main()
