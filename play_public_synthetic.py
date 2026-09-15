import argparse
import asyncio
import functools
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from poke_env.ps_client.server_configuration import ServerConfiguration

import metamon
from metamon.rl.pretrained import get_pretrained_model
from metamon.rl.metamon_to_amago import PSLadderAMAGOWrapper
from metamon.env import TeamSet
from metamon.env.wrappers import QueueOnLocalLadder
from battle_ai import BattleLogger, TacticalEvaluator, snapshot_battle


PUBLIC_SERVER = ServerConfiguration(
    "wss://sim3.psim.us/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)


class SessionPhase(str, Enum):
    IDLE = "idle"
    QUEUING = "queuing"
    CREATED = "created"
    ACTIVE = "active"
    TURN = "turn"
    ENDED = "ended"
    CLEANUP = "cleanup"


@dataclass
class BattleSessionState:
    """Explicit one-battle-at-a-time lifecycle state."""

    phase: SessionPhase = SessionPhase.IDLE
    battle_tag: str | None = None
    generation: int = 0

    def begin_queue(self):
        if self.phase not in {SessionPhase.IDLE, SessionPhase.CLEANUP}:
            raise RuntimeError(f"cannot queue from {self.phase.value}")
        self.generation += 1
        self.battle_tag = None
        self.phase = SessionPhase.QUEUING

    def battle_created(self, battle_tag):
        if self.phase not in {SessionPhase.QUEUING, SessionPhase.CREATED}:
            raise RuntimeError(f"cannot create from {self.phase.value}")
        if self.battle_tag is not None and self.battle_tag != battle_tag:
            raise RuntimeError(f"stale battle {self.battle_tag} cannot replace {battle_tag}")
        self.battle_tag = battle_tag
        self.phase = SessionPhase.CREATED

    def battle_started(self, battle_tag):
        if self.battle_tag != battle_tag or self.phase not in {SessionPhase.CREATED, SessionPhase.ACTIVE, SessionPhase.TURN}:
            raise RuntimeError(f"battle {battle_tag} is not active")
        self.phase = SessionPhase.ACTIVE

    def turn_received(self, battle_tag):
        if self.battle_tag != battle_tag or self.phase not in {SessionPhase.ACTIVE, SessionPhase.TURN}:
            raise RuntimeError(f"turn for stale battle {battle_tag}")
        self.phase = SessionPhase.TURN

    def battle_ended(self, battle_tag):
        if self.battle_tag != battle_tag or self.phase not in {SessionPhase.ACTIVE, SessionPhase.TURN, SessionPhase.ENDED}:
            raise RuntimeError(f"cannot end inactive battle {battle_tag}")
        self.phase = SessionPhase.ENDED

    def cleanup_started(self):
        if self.phase != SessionPhase.ENDED:
            raise RuntimeError(f"cannot clean up from {self.phase.value}")
        self.phase = SessionPhase.CLEANUP

    def queue_cancelled(self):
        if self.phase not in {SessionPhase.QUEUING, SessionPhase.CREATED}:
            raise RuntimeError(f"cannot cancel queue from {self.phase.value}")
        self.phase = SessionPhase.CLEANUP

    def cleanup_finished(self):
        if self.phase != SessionPhase.CLEANUP:
            raise RuntimeError(f"cleanup finished from {self.phase.value}")
        self.phase = SessionPhase.IDLE
        self.battle_tag = None


class TransportTraceWriter:
    """Redacted JSONL trace for the public battle transport pipeline."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def redact_text(value):
        if value is None:
            return None
        text = str(value)
        if text.startswith("/trn "):
            parts = text.split(",", 2)
            if len(parts) == 3:
                return f"{parts[0]},{parts[1]},[redacted-assertion]"
            return "/trn [redacted]"
        if text.startswith("/utm "):
            return "/utm [redacted-team]"
        if "|/trn " in text:
            room, message = text.split("|", 1)
            return room + "|" + TransportTraceWriter.redact_text(message)
        if "|/utm " in text:
            room, _ = text.split("|", 1)
            return room + "|/utm [redacted-team]"
        if "assertion" in text.lower():
            return "[redacted-auth-message]"
        return text

    @staticmethod
    def room_from_raw(message):
        text = str(message or "")
        first_line = text.splitlines()[0] if text else ""
        if first_line.startswith(">"):
            return first_line[1:]
        return ""

    @staticmethod
    def compact_battle_state(battle):
        if battle is None:
            return {}
        active = getattr(battle, "active_pokemon", None)
        opponent_active = getattr(battle, "opponent_active_pokemon", None)
        return {
            "battle_tag": str(getattr(battle, "battle_tag", "")),
            "turn": int(getattr(battle, "turn", 0) or 0),
            "finished": bool(getattr(battle, "finished", False)),
            "won": bool(getattr(battle, "won", False)),
            "lost": bool(getattr(battle, "lost", False)),
            "waiting": bool(getattr(battle, "_wait", False)),
            "trapped": bool(getattr(battle, "trapped", False)),
            "active": str(getattr(active, "species", "") or ""),
            "opponent_active": str(getattr(opponent_active, "species", "") or ""),
            "available_moves": [
                str(getattr(move, "id", move))
                for move in (getattr(battle, "available_moves", None) or [])
            ],
            "available_switches": [
                str(getattr(mon, "species", mon))
                for mon in (getattr(battle, "available_switches", None) or [])
            ],
        }

    def event(self, stage, battle_tag="", **fields):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "battle_tag": str(battle_tag or ""),
        }
        record.update(fields)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")


class PublicQueueOnLadder(QueueOnLocalLadder):
    matchmaking_timeout = 90

    def handle_ladder_start(self, n_challenges: int):
        self._install_transport_trace_hooks()
        return self.start_laddering(n_challenges=n_challenges)

    def start_laddering(self, n_challenges=None, callback=None, sleep_between=None):
        """Retry stale public searches while keeping one battle/search active."""
        from poke_env.concurrency import POKE_LOOP
        if getattr(self, "_challenge_task", None) is not None and not self._challenge_task.done():
            raise RuntimeError("Agent is already challenging")
        self._challenge_task = asyncio.run_coroutine_threadsafe(
            self._resilient_ladder_loop(n_challenges, callback, sleep_between), POKE_LOOP
        )

    async def _resilient_ladder_loop(self, n_challenges, callback, sleep_between):
        completed = 0
        while n_challenges is None or completed < n_challenges:
            print(f"[public ladder] searching for battle {completed + 1}/{n_challenges or 'infinite'}...", flush=True)
            self._session.begin_queue()
            self._trace("lifecycle.queue_begin", generation=self._session.generation)
            task = asyncio.create_task(self.agent.ladder(1))
            deadline = asyncio.get_running_loop().time() + self.matchmaking_timeout
            timed_out = False
            self._battle_done.clear()
            battle_started = False
            while not task.done():
                current = getattr(self.agent, "current_battle", None)
                # Once Showdown has assigned a live battle, never cancel or
                # retry this task: battle duration is independent of queue time.
                if current is not None and not getattr(current, "finished", True):
                    battle_started = True
                    tag = str(getattr(current, "battle_tag", ""))
                    if self._session.battle_tag is None:
                        self._session.battle_created(tag)
                        self._trace("lifecycle.battle_created", tag, state=TransportTraceWriter.compact_battle_state(current))
                    self._session.battle_started(tag)
                    self._trace("lifecycle.battle_active", tag, state=TransportTraceWriter.compact_battle_state(current))
                if not battle_started and asyncio.get_running_loop().time() >= deadline:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    timed_out = True
                    break
                if battle_started and self._battle_done.is_set():
                    break
                if battle_started:
                    self._maybe_trace_stall(task)
                await asyncio.sleep(1)
            if timed_out:
                self._session.queue_cancelled()
                self._session.cleanup_finished()
                self._trace("lifecycle.queue_timeout")
                print("[public ladder] matchmaking search timed out; retrying", flush=True)
                continue
            # The installed poke-env ladder coroutine can remain pending after
            # the battle has already been delivered to OpenAIGymEnv. Drive the
            # episode from the battle state, then release that stale waiter.
            while not self._battle_done.is_set():
                await asyncio.sleep(1)
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            # poke-env retains finished battles in its registry.  Starting the
            # next ladder coroutine before clearing that registry can make the
            # next environment reset act on the old room ("Battle is already
            # finished") and leave the sequential evaluator waiting forever.
            try:
                self._session.cleanup_started()
                self._trace("lifecycle.cleanup_started", self._session.battle_tag)
                self.agent.reset_battles()
            except EnvironmentError:
                # A genuinely live battle must never be discarded.  The loop
                # will retry only after poke-env reports it finished.
                await asyncio.sleep(1)
                self.agent.reset_battles()
            self._session.cleanup_finished()
            self._trace("lifecycle.cleanup_finished")
            completed += 1
            if callback and self.current_battle is not None:
                callback(self.current_battle)
            print(f"[public ladder] completed battle {completed}/{n_challenges or 'infinite'}", flush=True)
            if completed < (n_challenges or completed + 1) and sleep_between is not None:
                await asyncio.sleep(sleep_between)

    def __init__(
        self,
        *args,
        battle_ai=None,
        battle_logger=None,
        decision_debug=False,
        transport_trace=None,
        transport_stall_seconds=60.0,
        **kwargs,
    ):
        self.battle_ai = battle_ai
        self.battle_logger = battle_logger
        self.decision_debug = decision_debug
        self.transport_trace = transport_trace
        self.transport_stall_seconds = transport_stall_seconds
        self._last_transport_progress = datetime.now(timezone.utc)
        self._stall_reported_for = None
        self._battle_done = threading.Event()
        self._session = BattleSessionState()
        super().__init__(*args, **kwargs)
        self._install_transport_trace_hooks()

    @staticmethod
    def _battle_tag(battle):
        return str(getattr(battle, "battle_tag", ""))

    def _trace(self, stage, battle_tag="", **fields):
        if self.transport_trace is not None:
            self.transport_trace.event(stage, battle_tag, **fields)

    def _touch_transport_progress(self, battle_tag=""):
        self._last_transport_progress = datetime.now(timezone.utc)
        self._stall_reported_for = None
        self._trace("transport.progress", battle_tag)

    def _maybe_trace_stall(self, ladder_task):
        if self.transport_trace is None or self.transport_stall_seconds is None:
            return
        battle = getattr(self.agent, "current_battle", None)
        tag = self._battle_tag(battle)
        if not tag or self._stall_reported_for == tag:
            return
        elapsed = (datetime.now(timezone.utc) - self._last_transport_progress).total_seconds()
        if elapsed < self.transport_stall_seconds:
            return
        ps_client = getattr(self.agent, "ps_client", None)
        websocket = getattr(ps_client, "websocket", None)
        loop = asyncio.get_running_loop()
        tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
        self._trace(
            "transport.stall_snapshot",
            tag,
            seconds_since_progress=elapsed,
            session_phase=self._session.phase.value,
            ladder_task_done=bool(ladder_task.done()),
            websocket_present=websocket is not None,
            websocket_closed=bool(getattr(websocket, "closed", False)) if websocket is not None else None,
            action_queue_empty=bool(getattr(getattr(self.agent, "actions", None), "empty", lambda: True)()),
            observation_queue_empty=bool(getattr(getattr(self.agent, "observations", None), "empty", lambda: True)()),
            active_task_count=len(tasks),
            battle_registry=list(getattr(self.agent, "_battles", {}).keys()),
            state=TransportTraceWriter.compact_battle_state(battle),
        )
        self._stall_reported_for = tag

    def _install_transport_trace_hooks(self):
        if self.transport_trace is None:
            return
        ps_client = getattr(getattr(self, "agent", None), "ps_client", None)
        if ps_client is None or getattr(ps_client, "_metamon_transport_trace_installed", False):
            return

        original_send_message = ps_client.send_message
        original_handle_message = ps_client._handle_message
        env = self

        async def traced_send_message(message, room="", message_2=None):
            safe_message = TransportTraceWriter.redact_text(message)
            env._trace(
                "websocket.outbound.before_send",
                room,
                room=str(room or ""),
                message=safe_message,
                message_2=TransportTraceWriter.redact_text(message_2),
            )
            try:
                result = await original_send_message(message, room, message_2)
            except Exception as exc:
                env._trace(
                    "websocket.outbound.error",
                    room,
                    room=str(room or ""),
                    message=safe_message,
                    error=repr(exc),
                )
                raise
            env._trace(
                "websocket.outbound.after_send",
                room,
                room=str(room or ""),
                message=safe_message,
            )
            env._touch_transport_progress(room)
            return result

        async def traced_handle_message(message):
            room = TransportTraceWriter.room_from_raw(message)
            env._trace(
                "websocket.inbound.raw",
                room,
                room=room,
                message=TransportTraceWriter.redact_text(message),
            )
            env._touch_transport_progress(room)
            return await original_handle_message(message)

        ps_client.send_message = traced_send_message
        ps_client._handle_message = traced_handle_message
        ps_client._metamon_transport_trace_installed = True

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        battle = self.current_battle
        tag = self._battle_tag(battle)
        if self._session.battle_tag is None:
            self._session.battle_created(tag)
            self._trace("lifecycle.battle_created_by_reset", tag, state=TransportTraceWriter.compact_battle_state(battle))
        self._session.battle_started(tag)
        self._trace("lifecycle.reset_ready", tag, state=TransportTraceWriter.compact_battle_state(battle))
        self._battle_done.clear()
        return result

    def action_to_move(self, action, battle):
        tag = self._battle_tag(battle)
        if battle is None or battle is not self.current_battle or getattr(battle, "finished", False):
            raise RuntimeError(f"refusing action for stale or finished battle {tag}")
        if self._session.battle_tag != tag:
            raise RuntimeError(f"refusing action for stale battle {tag}; active={self._session.battle_tag}")
        self._session.turn_received(tag)
        model_action = int(action)
        final_action = model_action
        candidates = []
        reasoning = {}
        self._trace(
            "action.model_selected",
            tag,
            model_action=model_action,
            legal_actions=list(getattr(self, "_most_recent_legal_actions", [])),
            state=TransportTraceWriter.compact_battle_state(battle),
        )
        if self.battle_ai is not None:
            state = snapshot_battle(battle)
            # PokeEnvWrapper has already computed this using the installed
            # UniversalState/action-space implementation before action_to_move.
            legal = list(self._most_recent_legal_actions)
            final_action, evaluations = self.battle_ai.evaluate(battle, legal, model_action)
            candidates = [e.__dict__ for e in evaluations]
            from battle_ai.insights import position_metadata
            reasoning = {"model_action": model_action, "selected": final_action,
                         "turn": state.turn, "format": state.format,
                         "position": position_metadata(state)}
            if self.decision_debug:
                print(f"\nTURN {state.turn}")
                print(f"OUR: {state.our_active.get('name')} HP {state.our_active.get('hp_fraction', 0.0):.1%} status={state.our_active.get('status')} boosts={state.our_active.get('boosts')}")
                print(f"OPPONENT: {state.opponent_active.get('name')} HP {state.opponent_active.get('hp_fraction', 0.0):.1%} status={state.opponent_active.get('status')} boosts={state.opponent_active.get('boosts')}")
                print(f"LEGAL ACTIONS: {legal}")
                print(f"SYNTHETICRLV2: selected={model_action}")
                for evaluation in evaluations:
                    print(f"  {evaluation.action}: {evaluation.label} kind={evaluation.kind} score={evaluation.tactical_score:.3f} KO={evaluation.ko_probability:.0%} | {evaluation.reason}")
                print(f"FINAL: {final_action} ({'model preserved' if final_action == model_action else 'verifier override'})")
            if self.battle_logger is not None:
                self.battle_logger.start(state.battle_id, username=state.player,
                                         opponent=state.opponent, format=state.format,
                                         team_file=str(getattr(self.metamon_team_set, "most_recent_team_file", "")))
                self.battle_logger.turn(state.battle_id, state.turn, state.to_dict(),
                                        candidates, model_action, final_action, reasoning)
        self._trace(
            "action.final_selected",
            tag,
            model_action=model_action,
            final_action=final_action,
            override=final_action != model_action,
        )
        order = super().action_to_move(final_action, battle)
        self._trace(
            "action.converted_to_order",
            tag,
            final_action=final_action,
            order_type=type(order).__name__,
            command=TransportTraceWriter.redact_text(getattr(order, "message", "")),
            state=TransportTraceWriter.compact_battle_state(battle),
        )
        return order

    def step(self, action):
        self._trace(
            "env.step.before",
            self._battle_tag(self.current_battle),
            action=int(action) if isinstance(action, (int, float)) else str(action),
            state=TransportTraceWriter.compact_battle_state(self.current_battle),
        )
        result = super().step(action)
        self._trace(
            "env.step.after",
            self._battle_tag(self.current_battle),
            action=int(action) if isinstance(action, (int, float)) else str(action),
            state=TransportTraceWriter.compact_battle_state(self.current_battle),
        )
        if self.battle_logger is not None:
            _, _, terminated, truncated, _ = result
            if bool(terminated) or bool(truncated):
                self._battle_done.set()
                battle = self.current_battle
                self._session.battle_ended(self._battle_tag(battle))
                self._trace("lifecycle.battle_ended", self._battle_tag(battle), state=TransportTraceWriter.compact_battle_state(battle))
                battle_id = str(getattr(battle, "battle_tag", "unknown"))
                won = bool(getattr(battle, "won", False))
                self.battle_logger.finish(
                    battle_id,
                    opponent_lead=str(getattr(getattr(battle, "opponent_active_pokemon", None), "species", "")),
                    winner=str(getattr(battle, "player_username", "")) if won else str(getattr(battle, "opponent_username", "")),
                    result="WIN" if won else "LOSS",
                    final_turn=int(getattr(battle, "turn", 0) or 0),
                )
        return result
    @property
    def server_configuration(self):
        return PUBLIC_SERVER


def make_public_ladder_env(
    battle_format,
    num_battles,
    observation_space,
    action_space,
    reward_function,
    player_username,
    player_password,
    player_team_set,
    battle_ai=None,
    battle_logger=None,
    decision_debug=False,
    transport_trace=None,
    transport_stall_seconds=60.0,
):
    env = PublicQueueOnLadder(
        battle_format=battle_format,
        observation_space=observation_space,
        action_space=action_space,
        reward_function=reward_function,
        num_battles=num_battles,
        player_username=player_username,
        player_password=player_password,
        player_team_set=player_team_set,
        battle_backend="poke-env",
        battle_ai=battle_ai,
        battle_logger=battle_logger,
        decision_debug=decision_debug,
        transport_trace=transport_trace,
        transport_stall_seconds=transport_stall_seconds,
    )

    return PSLadderAMAGOWrapper(env)


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)

    parser.add_argument(
        "--team_dir",
        default="public_gen3ou_teams",
    )
    parser.add_argument("--team-dir", dest="team_dir_alias", default=None)

    parser.add_argument(
        "--battles",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--checkpoint",
        type=int,
        default=None,
    )
    parser.add_argument("--enable-battle-ai", action="store_true")
    parser.add_argument("--log-dir", default="battle_data")
    parser.add_argument("--database", default=None)
    parser.add_argument("--analysis", action="store_true", help="Print a summary after the run")
    parser.add_argument("--config", default=None, help="JSON-compatible YAML tactical config")
    parser.add_argument("--decision-debug", action="store_true")
    parser.add_argument("--transport-debug", action="store_true", help="Write redacted websocket/action pipeline trace")
    parser.add_argument("--transport-trace", default=None, help="JSONL path for --transport-debug")
    parser.add_argument("--transport-stall-seconds", type=float, default=60.0)

    args = parser.parse_args()

    battle_format = "gen3ou"
    team_dir = Path(args.team_dir_alias or args.team_dir)

    if not team_dir.exists():
        raise FileNotFoundError(
            f"Team directory not found: {team_dir}"
        )

    team_files = sorted(team_dir.glob("*.gen3ou_team"))

    if not team_files:
        raise ValueError(
            f"No .gen3ou_team files found in {team_dir}"
        )

    print()
    print("========================================")
    print(" SyntheticRLV2 Public Gen 3 OU")
    print("========================================")
    print(f"Username:    {args.username}")
    print(f"Format:      {battle_format}")
    print(f"Battles:     {args.battles}")
    print(f"Team pool:   {team_dir}")
    print(f"Team count:  {len(team_files)}")
    print(f"Temperature: {args.temperature}")
    print()

    print("Team pool:")
    for team_file in team_files:
        print(f"  {team_file.name}")

    print()

    # Metamon TeamSet selects a team from this directory for each episode.
    player_team_set = TeamSet(
        str(team_dir),
        battle_format,
    )

    pretrained_model = get_pretrained_model("SyntheticRLV2")

    checkpoint = (
        args.checkpoint
        if args.checkpoint is not None
        else pretrained_model.default_checkpoint
    )

    print("Loading SyntheticRLV2...")
    print(f"Checkpoint: {checkpoint}")

    agent = pretrained_model.initialize_agent(
        checkpoint=checkpoint,
        log=False,
        action_temperature=args.temperature,
    )

    # IMPORTANT: one environment/one battle at a time.
    agent.env_mode = "sync"
    agent.parallel_actors = 1
    agent.verbose = False

    make_env = functools.partial(
        make_public_ladder_env,
        battle_format=battle_format,
        num_battles=args.battles,
        observation_space=pretrained_model.observation_space,
        action_space=pretrained_model.action_space,
        reward_function=pretrained_model.reward_function,
        player_username=args.username,
        player_password=args.password,
        player_team_set=player_team_set,
    )

    logger = None
    tactical = None
    transport_trace = None
    if args.transport_debug:
        trace_path = args.transport_trace or str(Path(args.log_dir) / "transport_trace.jsonl")
        transport_trace = TransportTraceWriter(trace_path)
        print(f"Transport trace: {trace_path}")
    if args.enable_battle_ai:
        database = args.database or str(Path(args.log_dir) / "battles.db")
        logger = BattleLogger(database)
        config_path = Path(args.config) if args.config else Path(__file__).parent / "battle_ai" / "config.yaml"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        tactical = TacticalEvaluator(**{k: config[k] for k in ("model_weight", "damage_weight", "ko_weight", "switch_penalty", "anti_throw_penalty", "override_mode") if k in config})
    if args.enable_battle_ai or args.decision_debug or transport_trace is not None:
        make_env = functools.partial(
            make_env,
            battle_ai=tactical,
            battle_logger=logger,
            decision_debug=args.decision_debug,
            transport_trace=transport_trace,
            transport_stall_seconds=args.transport_stall_seconds,
        )

    print()
    print("Connecting SyntheticRLV2 to public Showdown...")
    print("Searching Gen 3 OU...")
    print()
    print(f"Running {args.battles} sequential battles...")
    print()

    results = agent.evaluate_test(
        [make_env],
        timesteps=args.battles * 1000,
        episodes=args.battles,
    )

    print()
    print("========================================")
    print(" RESULTS")
    print("========================================")
    print(results)
    if logger is not None:
        logger.close()
    if args.analysis:
        import subprocess, sys
        database = args.database or str(Path(args.log_dir) / "battles.db")
        subprocess.run([sys.executable, "analyze_battles.py", "--database", database, "--last", str(args.battles)], check=False)


if __name__ == "__main__":
    asyncio.run(main())
