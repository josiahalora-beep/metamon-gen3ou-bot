"""Authoritative asymmetric local evaluation for Level 5 champion gating.

This runner starts one challenger and one acceptor process per worker, gives each
process its own SQLite BattleLogger database, and verifies the two logs agree on
the completed battles and winners before producing a successful result artifact.

It intentionally fails closed: missing DBs, missing battle rows, mismatched logs,
ambiguous winners, worker failures, and incomplete battle counts cannot produce a
successful evaluation.
"""
from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import subprocess
import sys
import time
import uuid
import sqlite3
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import play_local_selfplay_synthetic_v2 as impl
import play_public_synthetic as runner
from metamon.rl.pretrained import LocalFinetunedModel

logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Level 5 authoritative asymmetric battle orchestrator")

    parser.add_argument("--challenger-type", choices=["checkpoint", "local"], required=True)
    parser.add_argument("--challenger-checkpoint", type=int)
    parser.add_argument("--challenger-run-dir")
    parser.add_argument("--challenger-run-name")
    parser.add_argument("--challenger-local-checkpoint", type=int)
    parser.add_argument("--challenger-name", default="Challenger")

    parser.add_argument("--acceptor-type", choices=["checkpoint", "local"], required=True)
    parser.add_argument("--acceptor-checkpoint", type=int)
    parser.add_argument("--acceptor-run-dir")
    parser.add_argument("--acceptor-run-name")
    parser.add_argument("--acceptor-local-checkpoint", type=int)
    parser.add_argument("--acceptor-name", default="Acceptor")

    parser.add_argument("--battles", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--team-dir", required=True)
    parser.add_argument("--evaluation-root", default="evaluation")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--decision-debug", action="store_true")
    return parser.parse_args()


def validate_model_args(prefix, args):
    kind = getattr(args, f"{prefix}_type")
    if kind == "checkpoint":
        checkpoint = getattr(args, f"{prefix}_checkpoint")
        if checkpoint is None:
            raise ValueError(f"Missing --{prefix}-checkpoint")
    else:
        required = [
            getattr(args, f"{prefix}_run_dir"),
            getattr(args, f"{prefix}_run_name"),
            getattr(args, f"{prefix}_local_checkpoint"),
        ]
        if any(value is None for value in required):
            raise ValueError(f"Local model requires --{prefix}-run-dir, --{prefix}-run-name, and --{prefix}-local-checkpoint")


def build_model(prefix, args):
    base = runner.get_pretrained_model("SyntheticRLV2")
    kind = getattr(args, f"{prefix}_type")
    if kind == "checkpoint":
        checkpoint = getattr(args, f"{prefix}_checkpoint")
        return base, checkpoint, f"SyntheticRLV2@{checkpoint}"

    run_dir = getattr(args, f"{prefix}_run_dir")
    run_name = getattr(args, f"{prefix}_run_name")
    local_checkpoint = getattr(args, f"{prefix}_local_checkpoint")
    model = LocalFinetunedModel(
        base_model=type(base),
        amago_ckpt_dir=str(Path(run_dir)),
        model_name=run_name,
        default_checkpoint=local_checkpoint,
    )
    return model, local_checkpoint, f"local:{run_name}@{local_checkpoint}"


def child_main(args):
    role = args.child_role
    prefix = "challenger" if role == "challenger" else "acceptor"
    username = args.username
    opponent_username = args.opponent_username
    model, checkpoint, model_label = build_model(prefix, args)

    team_dir = Path(args.team_dir)
    if not team_dir.exists() or not list(team_dir.rglob("*.gen3ou_team")):
        raise FileNotFoundError(f"No Gen 3 OU teams found under {team_dir}")

    log_dir = Path(args.log_dir)
    trajectory_dir = Path(args.trajectory_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    player_team_set = runner.TeamSet(str(team_dir), "gen3ou")
    print(f"[{role}] Loading {model_label} as {username}...", flush=True)
    agent = model.initialize_agent(
        checkpoint=checkpoint,
        log=False,
        action_temperature=args.temperature,
    )
    agent.env_mode = "sync"
    agent.parallel_actors = 1
    agent.verbose = False

    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger = runner.BattleLogger(str(db_path))

    tactical = None
    config_path = Path(args.config) if args.config else REPO_ROOT / "battle_ai" / "config.yaml"
    if config_path.exists():
        try:
            import yaml
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except ImportError:
            config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        config = {}
    tactical = runner.TacticalEvaluator(**{
        key: config[key]
        for key in (
            "model_weight", "damage_weight", "ko_weight", "switch_penalty",
            "anti_throw_penalty", "override_mode",
        )
        if key in config
    })

    make_env = functools.partial(
        impl.make_local_challenge_env,
        battle_format="gen3ou",
        num_battles=args.battles,
        observation_space=model.observation_space,
        action_space=model.action_space,
        reward_function=model.reward_function,
        player_username=username,
        opponent_username=opponent_username,
        role=role,
        player_team_set=player_team_set,
        save_trajectories_to=str(trajectory_dir / role),
        battle_ai=tactical,
        battle_logger=logger,
        decision_debug=args.decision_debug,
        transport_trace=None,
        transport_stall_seconds=60.0,
    )

    try:
        print(f"[{role}] {username} vs {opponent_username}: {args.battles} battles", flush=True)
        results = agent.evaluate_test(
            [make_env],
            timesteps=args.battles * 1000,
            episodes=args.battles,
        )
        print(f"[{role}] RESULTS: {results}", flush=True)
    finally:
        logger.close()
    return 0


def build_child_command(args, role, username, opponent_username, worker_dir, battles):
    prefix = "challenger" if role == "challenger" else "acceptor"
    log_dir = worker_dir / role / "logs"
    trajectory_dir = worker_dir / role / "trajectories"
    db_path = worker_dir / role / "battles.db"
    log_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir.mkdir(parents=True, exist_ok=True)

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
        "--db-path", str(db_path),
        "--config", args.config or "",
    ]
    for name in ("challenger", "acceptor"):
        kind = getattr(args, f"{name}_type")
        if name != prefix:
            continue
        if kind == "checkpoint":
            cmd += [f"--{name}-checkpoint", str(getattr(args, f"{name}_checkpoint"))]
        else:
            cmd += [
                f"--{name}-run-dir", str(getattr(args, f"{name}_run_dir")),
                f"--{name}-run-name", str(getattr(args, f"{name}_run_name")),
                f"--{name}-local-checkpoint", str(getattr(args, f"{name}_local_checkpoint")),
            ]
    if args.decision_debug:
        cmd.append("--decision-debug")
    return cmd


def launch_worker(args, worker_id, battles, eval_dir):
    uid = uuid.uuid4().hex[:8]
    acc_user = f"Level5A{worker_id}{uid}"
    cha_user = f"Level5C{worker_id}{uid}"
    worker_dir = eval_dir / "workers" / f"worker_{worker_id:02d}"
    worker_dir.mkdir(parents=True, exist_ok=True)

    acc_cmd = build_child_command(args, "acceptor", acc_user, cha_user, worker_dir, battles)
    cha_cmd = build_child_command(args, "challenger", cha_user, acc_user, worker_dir, battles)
    acc_log = open(worker_dir / "acceptor" / "process.log", "w", encoding="utf-8")
    cha_log = open(worker_dir / "challenger" / "process.log", "w", encoding="utf-8")

    logging.info(f"[Worker {worker_id:02d}] Launching {battles} battles: {cha_user} vs {acc_user}")
    p_acc = subprocess.Popen(acc_cmd, cwd=REPO_ROOT, stdout=acc_log, stderr=subprocess.STDOUT, text=True)
    time.sleep(2.0)
    p_cha = subprocess.Popen(cha_cmd, cwd=REPO_ROOT, stdout=cha_log, stderr=subprocess.STDOUT, text=True)
    return {
        "id": worker_id,
        "battles": battles,
        "acc": p_acc,
        "cha": p_cha,
        "acc_log": acc_log,
        "cha_log": cha_log,
        "acc_user": acc_user,
        "cha_user": cha_user,
        "acc_db": worker_dir / "acceptor" / "battles.db",
        "cha_db": worker_dir / "challenger" / "battles.db",
    }


def read_db(path: Path, expected_pair):
    if not path.exists():
        raise RuntimeError(f"Missing authoritative database: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='battles'"
        ).fetchone()
        if table is None:
            raise RuntimeError(f"Missing battles table: {path}")
        rows = conn.execute(
            "SELECT battle_id, username, opponent, winner, result, finished_at FROM battles"
        ).fetchall()
    except sqlite3.Error as exc:
        raise RuntimeError(f"SQLite verification error in {path}: {exc}") from exc
    finally:
        conn.close()

    records = {}
    allowed = set(expected_pair)
    for row in rows:
        battle_id = str(row["battle_id"])
        username = str(row["username"] or "")
        opponent = str(row["opponent"] or "")
        winner = str(row["winner"] or "")
        result = str(row["result"] or "")
        if {username, opponent} != allowed:
            continue
        if not row["finished_at"]:
            continue
        if winner not in allowed:
            raise RuntimeError(f"Ambiguous winner '{winner}' for battle {battle_id} in {path}")
        if battle_id in records:
            raise RuntimeError(f"Duplicate battle_id '{battle_id}' in {path}")
        records[battle_id] = {
            "username": username,
            "opponent": opponent,
            "winner": winner,
            "result": result,
        }
    return records


def verify_workers(workers):
    total = 0
    challenger_wins = 0
    acceptor_wins = 0
    draws = 0
    per_worker = []

    for worker in workers:
        pair = (worker["cha_user"], worker["acc_user"])
        acc = read_db(worker["acc_db"], pair)
        cha = read_db(worker["cha_db"], pair)
        if set(acc) != set(cha):
            raise RuntimeError(
                f"Worker {worker['id']:02d} log mismatch: "
                f"acceptor={len(acc)} challenger={len(cha)}"
            )
        for battle_id in acc:
            a = acc[battle_id]
            c = cha[battle_id]
            if a["winner"] != c["winner"]:
                raise RuntimeError(f"Worker {worker['id']:02d} winner mismatch for {battle_id}")
            winner = a["winner"]
            if winner == worker["cha_user"]:
                challenger_wins += 1
            elif winner == worker["acc_user"]:
                acceptor_wins += 1
            else:
                raise RuntimeError(f"Unexpected winner {winner} for {battle_id}")
        total += len(acc)
        per_worker.append({
            "worker": worker["id"],
            "requested": worker["battles"],
            "completed": len(acc),
            "challenger_wins": sum(1 for r in acc.values() if r["winner"] == worker["cha_user"]),
            "acceptor_wins": sum(1 for r in acc.values() if r["winner"] == worker["acc_user"]),
        })

    if total != challenger_wins + acceptor_wins + draws:
        raise RuntimeError("Outcome summation invariant violated")
    return total, challenger_wins, acceptor_wins, draws, per_worker


def main():
    args = parse_args()
    if getattr(args, "child_mode", False):
        return child_main(args)

    if args.battles <= 0:
        raise ValueError("--battles must be greater than zero")
    if args.workers <= 0:
        raise ValueError("--workers must be greater than zero")
    args.workers = min(args.workers, args.battles)
    validate_model_args("challenger", args)
    validate_model_args("acceptor", args)

    eval_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    eval_dir = Path(args.evaluation_root) / eval_id
    eval_dir.mkdir(parents=True, exist_ok=True)
    metadata = vars(args).copy()
    metadata["evaluation_id"] = eval_id
    (eval_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    logging.info("========================================")
    logging.info(f" AUTHORITATIVE LEVEL 5 EVALUATION [{eval_id}]")
    logging.info("========================================")
    logging.info(f"Requested battles: {args.battles} | Workers: {args.workers}")

    base = args.battles // args.workers
    remainder = args.battles % args.workers
    workers = []
    failed = False
    failure_reason = ""
    start = time.monotonic()

    try:
        for worker_id in range(args.workers):
            count = base + (1 if worker_id < remainder else 0)
            workers.append(launch_worker(args, worker_id, count, eval_dir))

        while True:
            all_done = all(w["acc"].poll() is not None and w["cha"].poll() is not None for w in workers)
            bad = next((w for w in workers if w["acc"].poll() not in (None, 0) or w["cha"].poll() not in (None, 0)), None)
            if bad is not None:
                failed = True
                failure_reason = f"Worker {bad['id']:02d} process failure: acceptor={bad['acc'].returncode}, challenger={bad['cha'].returncode}"
                break
            if all_done:
                break
            if time.monotonic() - start >= args.timeout_seconds:
                failed = True
                failure_reason = f"Global timeout after {args.timeout_seconds}s"
                break
            time.sleep(1.0)
    except Exception as exc:
        failed = True
        failure_reason = f"Orchestrator exception: {exc}"
    finally:
        if failed:
            for w in workers:
                for proc in (w["acc"], w["cha"]):
                    if proc.poll() is None:
                        proc.kill()
        for w in workers:
            for proc in (w["acc"], w["cha"]):
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            w["acc_log"].close()
            w["cha_log"].close()

    try:
        completed, cha_wins, acc_wins, draws, per_worker = verify_workers(workers)
    except Exception as exc:
        completed = 0
        cha_wins = acc_wins = draws = 0
        per_worker = []
        if not failure_reason:
            failure_reason = str(exc)
        failed = True

    complete = (not failed) and completed == args.battles
    result = {
        "evaluation_id": eval_id,
        "challenger": args.challenger_name,
        "challenger_model": (
            f"SyntheticRLV2@{args.challenger_checkpoint}"
            if args.challenger_type == "checkpoint"
            else f"local:{args.challenger_run_name}@{args.challenger_local_checkpoint}"
        ),
        "acceptor": args.acceptor_name,
        "acceptor_model": (
            f"SyntheticRLV2@{args.acceptor_checkpoint}"
            if args.acceptor_type == "checkpoint"
            else f"local:{args.acceptor_run_name}@{args.acceptor_local_checkpoint}"
        ),
        "battles_requested": args.battles,
        "battles_completed": completed,
        "challenger_wins": cha_wins,
        "acceptor_wins": acc_wins,
        "draws": draws,
        "challenger_win_rate": (cha_wins / completed) if completed else 0.0,
        "status": "complete" if complete else "incomplete",
        "per_worker": per_worker,
    }
    if failure_reason:
        result["error"] = failure_reason

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if complete:
        logging.info(f"Evaluation verified: {completed}/{args.battles} battles")
        logging.info(f"Challenger win rate: {result['challenger_win_rate']:.1%}")
        return 0
    logging.error(f"Evaluation FAILED CLOSED: {completed}/{args.battles} verified battles")
    if failure_reason:
        logging.error(f"Reason: {failure_reason}")
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    # Child dispatch uses the real parser so all normal flags remain documented.
    # We detect child mode before normal parsing without requiring every parent flag.
    if "--child-mode" in sys.argv:
        # Re-enter main with a parser extended for the child-only fields.
        pass
    try:
        sys.exit(main())
    except Exception as exc:
        logging.error(f"❌ {exc}")
        sys.exit(1)
