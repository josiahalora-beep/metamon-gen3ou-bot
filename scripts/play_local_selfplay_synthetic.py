"""Run SyntheticRLV2 against a second SyntheticRLV2 locally.

This deliberately avoids the local ladder. Two independent AMAGO processes connect to
one accelerated Showdown instance and play deterministic head-to-head challenges.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import play_public_synthetic as runner
from metamon.env.wrappers import ChallengeByUsername
from metamon.rl.metamon_to_amago import PSLadderAMAGOWrapper
from poke_env.ps_client.server_configuration import ServerConfiguration

LOCAL_SERVER = ServerConfiguration(
    "ws://127.0.0.1:8000/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)
runner.PUBLIC_SERVER = LOCAL_SERVER


class LocalSyntheticChallenge(ChallengeByUsername):
    """Challenge-mode environment reusing the existing battle verifier/logger."""

    def __init__(self, *args, battle_ai=None, battle_logger=None, decision_debug=False,
                 transport_trace=None, transport_stall_seconds=60.0, **kwargs):
        self.battle_ai = battle_ai
        self.battle_logger = battle_logger
        self.decision_debug = decision_debug
        self.transport_trace = transport_trace
        self.transport_stall_seconds = transport_stall_seconds
        self._last_transport_progress = datetime.now(timezone.utc)
        self._stall_reported_for = None
        self._battle_done = threading.Event()
        self._session = runner.BattleSessionState()
        super().__init__(*args, **kwargs)
        self._install_transport_trace_hooks()

    @property
    def server_configuration(self):
        return LOCAL_SERVER

    _install_transport_trace_hooks = runner.PublicQueueOnLadder._install_transport_trace_hooks
    _trace = runner.PublicQueueOnLadder._trace
    _touch_transport_progress = runner.PublicQueueOnLadder._touch_transport_progress
    _maybe_trace_stall = runner.PublicQueueOnLadder._maybe_trace_stall

    def reset(self, *args, **kwargs):
        if self._session.phase == runner.SessionPhase.ENDED:
            self._session.cleanup_started()
            self._session.cleanup_finished()
        result = super().reset(*args, **kwargs)
        battle = self.current_battle
        tag = self._battle_tag(battle)
        if self._session.battle_tag is None:
            self._session.battle_created(tag)
            self._trace("lifecycle.battle_created_by_reset", tag,
                         state=runner.TransportTraceWriter.compact_battle_state(battle))
        self._session.battle_started(tag)
        self._trace("lifecycle.reset_ready", tag,
                     state=runner.TransportTraceWriter.compact_battle_state(battle))
        self._battle_done.clear()
        return result

    @staticmethod
    def _battle_tag(battle):
        return str(getattr(battle, "battle_tag", ""))

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
            state=runner.TransportTraceWriter.compact_battle_state(battle),
        )
        if self.battle_ai is not None:
            state = runner.snapshot_battle(battle)
            legal = list(self._most_recent_legal_actions)
            final_action, evaluations = self.battle_ai.evaluate(battle, legal, model_action)
            candidates = [e.__dict__ for e in evaluations]
            from battle_ai.insights import position_metadata
            reasoning = {
                "model_action": model_action,
                "selected": final_action,
                "turn": state.turn,
                "format": state.format,
                "position": position_metadata(state),
            }
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
                self.battle_logger.start(
                    state.battle_id,
                    username=state.player,
                    opponent=state.opponent,
                    format=state.format,
                    team_file=str(getattr(self.metamon_team_set, "most_recent_team_file", "")),
                )
                self.battle_logger.turn(
                    state.battle_id, state.turn, state.to_dict(),
                    candidates, model_action, final_action, reasoning,
                )
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
            command=runner.TransportTraceWriter.redact_text(getattr(order, "message", "")),
            state=runner.TransportTraceWriter.compact_battle_state(battle),
        )
        return order

    def step(self, action):
        self._trace(
            "env.step.before",
            self._battle_tag(self.current_battle),
            action=int(action) if isinstance(action, (int, float)) else str(action),
            state=runner.TransportTraceWriter.compact_battle_state(self.current_battle),
        )
        result = super().step(action)
        self._trace(
            "env.step.after",
            self._battle_tag(self.current_battle),
            action=int(action) if isinstance(action, (int, float)) else str(action),
            state=runner.TransportTraceWriter.compact_battle_state(self.current_battle),
        )
        if self.battle_logger is not None:
            _, _, terminated, truncated, _ = result
            if bool(terminated) or bool(truncated):
                self._battle_done.set()
                battle = self.current_battle
                self._session.battle_ended(self._battle_tag(battle))
                self._trace("lifecycle.battle_ended", self._battle_tag(battle),
                             state=runner.TransportTraceWriter.compact_battle_state(battle))
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


def make_local_challenge_env(
    battle_format,
    num_battles,
    observation_space,
    action_space,
    reward_function,
    player_username,
    opponent_username,
    role,
    player_team_set,
    save_trajectories_to=None,
    battle_ai=None,
    battle_logger=None,
    decision_debug=False,
    transport_trace=None,
    transport_stall_seconds=60.0,
):
    env = LocalSyntheticChallenge(
        battle_format=battle_format,
        num_battles=num_battles,
        observation_space=observation_space,
        action_space=action_space,
        reward_function=reward_function,
        player_team_set=player_team_set,
        player_username=player_username,
        opponent_username=opponent_username,
        role=role,
        battle_backend="poke-env",
        save_trajectories_to=save_trajectories_to,
        battle_ai=battle_ai,
        battle_logger=battle_logger,
        decision_debug=decision_debug,
        transport_trace=transport_trace,
        transport_stall_seconds=transport_stall_seconds,
    )
    return PSLadderAMAGOWrapper(env)


def run_role(args, role: str, username: str, opponent_username: str, log_dir: Path) -> int:
    battle_format = "gen3ou"
    team_dir = Path(args.team_dir)
    if not team_dir.exists():
        raise FileNotFoundError(f"Team directory not found: {team_dir}")
    if not sorted(team_dir.glob("*.gen3ou_team")):
        raise ValueError(f"No .gen3ou_team files found in {team_dir}")

    player_team_set = runner.TeamSet(str(team_dir), battle_format)
    pretrained_model = runner.get_pretrained_model("SyntheticRLV2")
    checkpoint = args.checkpoint if args.checkpoint is not None else pretrained_model.default_checkpoint

    print(f"[{role}] Loading SyntheticRLV2 checkpoint {checkpoint} as {username}...", flush=True)
    agent = pretrained_model.initialize_agent(
        checkpoint=checkpoint,
        log=False,
        action_temperature=args.temperature,
    )
    agent.env_mode = "sync"
    agent.parallel_actors = 1
    agent.verbose = False

    logger = runner.BattleLogger(str(log_dir / "battles.db")) if args.enable_battle_ai else None
    tactical = None
    if args.enable_battle_ai:
        config_path = Path(args.config) if args.config else Path(__file__).parent / "battle_ai" / "config.yaml"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        tactical = runner.TacticalEvaluator(**{
            k: config[k] for k in (
                "model_weight", "damage_weight", "ko_weight", "switch_penalty",
                "anti_throw_penalty", "override_mode",
            ) if k in config
        })

    transport_trace = None
    if args.transport_debug:
        trace_path = args.transport_trace or str(log_dir / "transport_trace.jsonl")
        transport_trace = runner.TransportTraceWriter(trace_path)

    make_env = functools.partial(
        make_local_challenge_env,
        battle_format=battle_format,
        num_battles=args.battles,
        observation_space=pretrained_model.observation_space,
        action_space=pretrained_model.action_space,
        reward_function=pretrained_model.reward_function,
        player_username=username,
        opponent_username=opponent_username,
        role=role,
        player_team_set=player_team_set,
        save_trajectories_to=str(args.trajectory_dir / role),
        battle_ai=tactical,
        battle_logger=logger,
        decision_debug=args.decision_debug,
        transport_trace=transport_trace,
        transport_stall_seconds=args.transport_stall_seconds,
    )

    print(f"[{role}] Local self-play: {username} vs {opponent_username} for {args.battles} battles", flush=True)
    print(f"[{role}] AMAGO trajectories: {args.trajectory_dir / role / battle_format}", flush=True)
    results = agent.evaluate_test(
        [make_env],
        timesteps=args.battles * 1000,
        episodes=args.battles,
    )
    print(f"[{role}] RESULTS: {results}", flush=True)
    if logger is not None:
        logger.close()
    return 0


def child_command(args, role, username, opponent_username, log_dir):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-role", role,
        "--username", username,
        "--opponent-username", opponent_username,
        "--team-dir", args.team_dir,
        "--battles", str(args.battles),
        "--temperature", str(args.temperature),
        "--log-dir", str(log_dir),
        "--trajectory-dir", str(args.trajectory_dir),
        "--child-mode",
    ]
    if args.checkpoint is not None:
        cmd += ["--checkpoint", str(args.checkpoint)]
    if args.enable_battle_ai:
        cmd += ["--enable-battle-ai"]
    if args.decision_debug:
        cmd += ["--decision-debug"]
    if args.transport_debug:
        cmd += ["--transport-debug"]
        if args.transport_trace:
            cmd += ["--transport-trace", str(log_dir / "transport_trace.jsonl")]
    if args.config:
        cmd += ["--config", args.config]
    return cmd


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", default="LocalSynthetic-A")
    parser.add_argument("--opponent-username", default="LocalSynthetic-B")
    parser.add_argument("--team-dir", default="public_gen3ou_teams")
    parser.add_argument("--battles", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--enable-battle-ai", action="store_true")
    parser.add_argument("--log-dir", default="battle_data/local_fast")
    parser.add_argument("--trajectory-dir", type=Path, default=Path("battle_data/local_fast/trajectories"))
    parser.add_argument("--analysis", action="store_true")
    parser.add_argument("--config", default=None)
    parser.add_argument("--decision-debug", action="store_true")
    parser.add_argument("--transport-debug", action="store_true")
    parser.add_argument("--transport-trace", default=None)
    parser.add_argument("--transport-stall-seconds", type=float, default=60.0)
    parser.add_argument("--child-mode", action="store_true")
    parser.add_argument("--child-role", choices=["challenger", "acceptor"], default=None)
    args = parser.parse_args()

    args.trajectory_dir = Path(args.trajectory_dir)
    args.trajectory_dir.mkdir(parents=True, exist_ok=True)

    if args.child_mode:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        code = run_role(args, args.child_role, args.username, args.opponent_username, log_dir)
        raise SystemExit(code)

    base_log = Path(args.log_dir)
    base_log.mkdir(parents=True, exist_ok=True)
    challenger = args.username
    acceptor = args.opponent_username
    acceptor_dir = base_log / "selfplay_acceptor"
    challenger_dir = base_log / "selfplay_challenger"

    acceptor_proc = subprocess.Popen(
        child_command(args, "acceptor", acceptor, challenger, acceptor_dir),
        cwd=REPO_ROOT,
    )
    challenger_code = 1
    challenger_proc = None
    try:
        time.sleep(3.0)
        challenger_proc = subprocess.Popen(
            child_command(args, "challenger", challenger, acceptor, challenger_dir),
            cwd=REPO_ROOT,
        )
        challenger_code = challenger_proc.wait()
    finally:
        if challenger_proc is not None and challenger_proc.poll() is None:
            challenger_proc.terminate()
            challenger_proc.wait(timeout=10)
        if acceptor_proc.poll() is None:
            acceptor_proc.terminate()
            acceptor_proc.wait(timeout=10)

    if challenger_code != 0 or acceptor_proc.returncode != 0:
        raise SystemExit(
            f"Local self-play failed: challenger={challenger_code}, acceptor={acceptor_proc.returncode}"
        )
    print(f"Completed {args.battles} local SyntheticRLV2-vs-SyntheticRLV2 battles.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
