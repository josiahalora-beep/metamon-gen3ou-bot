"""Continuous local curriculum self-play runner.

Loads either the public SyntheticRLV2 checkpoint or a local finetune checkpoint,
then reuses the battle-safe local challenge environment. Multiple independent
self-play worker pairs can run concurrently against one accelerated Showdown
server. Each worker has isolated logs/trajectory output.
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import metamon.env.wrappers as metamon_wrappers
from datetime import datetime as _RealDateTime


class _SafeDateTime(_RealDateTime):
    def strftime(self, fmt):
        return super().strftime(fmt).replace(":", "-")


metamon_wrappers.datetime = _SafeDateTime

import play_local_selfplay_synthetic_v2 as impl
import play_public_synthetic as runner
from metamon.rl.pretrained import LocalFinetunedModel


def build_model(args):
    base = runner.get_pretrained_model("SyntheticRLV2")
    if args.local_run_dir:
        if not args.local_run_name or args.local_checkpoint is None:
            raise ValueError("local model requires --local-run-name and --local-checkpoint")
        model = LocalFinetunedModel(
            base_model=type(base),
            amago_ckpt_dir=str(Path(args.local_run_dir)),
            model_name=args.local_run_name,
            default_checkpoint=args.local_checkpoint,
        )
        return model, f"local:{args.local_run_name}@{args.local_checkpoint}"
    return base, f"SyntheticRLV2@{args.checkpoint if args.checkpoint is not None else base.default_checkpoint}"


def run_role(args, role, username, opponent_username, log_dir):
    battle_format = "gen3ou"
    team_dir = Path(args.team_dir)
    if not team_dir.exists() or not list(team_dir.rglob("*.gen3ou_team")):
        raise FileNotFoundError(f"No Gen 3 OU teams found under {team_dir}")

    player_team_set = runner.TeamSet(str(team_dir), battle_format)
    model, model_label = build_model(args)
    checkpoint = args.checkpoint if args.checkpoint is not None else model.default_checkpoint
    print(f"[{role}] Loading {model_label}...", flush=True)
    agent = model.initialize_agent(
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
        if config_path.exists():
            try:
                import yaml
                config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            except ImportError:
                config = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            config = {}
        tactical = runner.TacticalEvaluator(**{
            k: config[k] for k in (
                "model_weight", "damage_weight", "ko_weight", "switch_penalty",
                "anti_throw_penalty", "override_mode",
            ) if k in config
        })

    make_env = functools.partial(
        impl.make_local_challenge_env,
        battle_format=battle_format,
        num_battles=args.battles,
        observation_space=model.observation_space,
        action_space=model.action_space,
        reward_function=model.reward_function,
        player_username=username,
        opponent_username=opponent_username,
        role=role,
        player_team_set=player_team_set,
        save_trajectories_to=str(args.trajectory_dir / role),
        battle_ai=tactical,
        battle_logger=logger,
        decision_debug=args.decision_debug,
        transport_trace=None,
        transport_stall_seconds=args.transport_stall_seconds,
    )

    print(f"[{role}] {username} vs {opponent_username}: {args.battles} battles; teams randomized independently", flush=True)
    print(f"[{role}] Trajectories: {args.trajectory_dir / role / battle_format}", flush=True)
    results = agent.evaluate_test(
        [make_env],
        timesteps=args.battles * 1000,
        episodes=args.battles,
    )
    print(f"[{role}] RESULTS: {results}", flush=True)
    if logger is not None:
        logger.close()
    return 0


def child_command(args, role, username, opponent_username, log_dir, trajectory_dir, battles):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-mode",
        "--child-role", role,
        "--username", username,
        "--opponent-username", opponent_username,
        "--team-dir", args.team_dir,
        "--battles", str(battles),
        "--temperature", str(args.temperature),
        "--log-dir", str(log_dir),
        "--trajectory-dir", str(trajectory_dir),
        "--enable-battle-ai",
    ]
    if args.checkpoint is not None:
        cmd += ["--checkpoint", str(args.checkpoint)]
    if args.local_run_dir:
        cmd += ["--local-run-dir", str(args.local_run_dir), "--local-run-name", args.local_run_name, "--local-checkpoint", str(args.local_checkpoint)]
    if args.decision_debug:
        cmd += ["--decision-debug"]
    if args.config:
        cmd += ["--config", args.config]
    return cmd


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--username", default="Curriculum-A")
    p.add_argument("--opponent-username", default="Curriculum-B")
    p.add_argument("--team-dir", default="public_gen3ou_teams")
    p.add_argument("--battles", type=int, default=100)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--checkpoint", type=int, default=None)
    p.add_argument("--local-run-dir", default=None)
    p.add_argument("--local-run-name", default=None)
    p.add_argument("--local-checkpoint", type=int, default=None)
    p.add_argument("--enable-battle-ai", action="store_true")
    p.add_argument("--log-dir", default="battle_data/local_fast")
    p.add_argument("--trajectory-dir", type=Path, default=Path("battle_data/local_fast/trajectories"))
    p.add_argument("--config", default=None)
    p.add_argument("--decision-debug", action="store_true")
    p.add_argument("--transport-stall-seconds", type=float, default=60.0)
    p.add_argument("--child-mode", action="store_true")
    p.add_argument("--child-role", choices=["challenger", "acceptor"], default=None)
    args = p.parse_args()
    args.trajectory_dir = Path(args.trajectory_dir)
    args.trajectory_dir.mkdir(parents=True, exist_ok=True)

    if args.child_mode:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        raise SystemExit(run_role(args, args.child_role, args.username, args.opponent_username, log_dir))

    workers = max(1, min(args.workers, args.battles))
    if args.battles < 1:
        raise ValueError("--battles must be at least 1")
    print(f"Launching {workers} parallel self-play workers for {args.battles} total battles...", flush=True)

    base_log = Path(args.log_dir)
    base_log.mkdir(parents=True, exist_ok=True)
    worker_processes = []
    base_battles = args.battles // workers
    remainder = args.battles % workers

    try:
        for worker in range(workers):
            worker_battles = base_battles + (1 if worker < remainder else 0)
            worker_root = args.trajectory_dir / f"worker_{worker + 1:02d}"
            worker_log = base_log / f"worker_{worker + 1:02d}"
            worker_root.mkdir(parents=True, exist_ok=True)
            worker_log.mkdir(parents=True, exist_ok=True)
            a_user = f"{args.username}-W{worker + 1}"
            b_user = f"{args.opponent_username}-W{worker + 1}"
            acceptor = subprocess.Popen(
                child_command(args, "acceptor", b_user, a_user, worker_log / "acceptor", worker_root, worker_battles),
                cwd=REPO_ROOT,
            )
            worker_processes.append((worker + 1, "acceptor", acceptor))
            time.sleep(1.0)
            challenger = subprocess.Popen(
                child_command(args, "challenger", a_user, b_user, worker_log / "challenger", worker_root, worker_battles),
                cwd=REPO_ROOT,
            )
            worker_processes.append((worker + 1, "challenger", challenger))

        failures = []
        for worker_id, role, process in worker_processes:
            code = process.wait()
            if code != 0:
                failures.append(f"worker {worker_id} {role} exited {code}")
        if failures:
            raise SystemExit("Curriculum self-play failed: " + "; ".join(failures))
    finally:
        for _, _, process in worker_processes:
            if process.poll() is None:
                process.terminate()
        for _, _, process in worker_processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()


if __name__ == "__main__":
    asyncio.run(main())
