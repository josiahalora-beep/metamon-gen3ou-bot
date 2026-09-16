"""Authoritative asymmetric local evaluation for Level 5 champion gating."""
from __future__ import annotations

import argparse
import functools
import json
import logging
import sqlite3
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import play_local_selfplay_synthetic_v2 as impl
import play_public_synthetic as runner
from metamon.rl.pretrained import LocalFinetunedModel

logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_parent():
    p = argparse.ArgumentParser(description="Level 5 authoritative asymmetric battle evaluator")
    for side in ("challenger", "acceptor"):
        p.add_argument(f"--{side}-type", choices=["checkpoint", "local"], required=True)
        p.add_argument(f"--{side}-checkpoint", type=int)
        p.add_argument(f"--{side}-run-dir")
        p.add_argument(f"--{side}-run-name")
        p.add_argument(f"--{side}-local-checkpoint", type=int)
        p.add_argument(f"--{side}-name", default=side.title())
    p.add_argument("--battles", type=int, required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--team-dir", required=True)
    p.add_argument("--evaluation-root", default="evaluation")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--timeout-seconds", type=int, default=1800)
    p.add_argument("--config", default=None)
    p.add_argument("--output-json", required=True)
    p.add_argument("--decision-debug", action="store_true")
    return p.parse_args()


def parse_child():
    p = argparse.ArgumentParser(description="Internal Level 5 worker")
    p.add_argument("--child-mode", action="store_true")
    p.add_argument("--child-role", choices=["challenger", "acceptor"], required=True)
    p.add_argument("--username", required=True)
    p.add_argument("--opponent-username", required=True)
    p.add_argument("--team-dir", required=True)
    p.add_argument("--battles", type=int, required=True)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--trajectory-dir", required=True)
    p.add_argument("--db-path", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--decision-debug", action="store_true")
    for side in ("challenger", "acceptor"):
        p.add_argument(f"--{side}-checkpoint", type=int)
        p.add_argument(f"--{side}-run-dir")
        p.add_argument(f"--{side}-run-name")
        p.add_argument(f"--{side}-local-checkpoint", type=int)
    return p.parse_args()


def model_for(prefix, args):
    base = runner.get_pretrained_model("SyntheticRLV2")
    run_dir = getattr(args, f"{prefix}_run_dir")
    if run_dir:
        run_name = getattr(args, f"{prefix}_run_name")
        local_checkpoint = getattr(args, f"{prefix}_local_checkpoint")
        if run_name is None or local_checkpoint is None:
            raise ValueError(f"{prefix} local model is missing run-name/checkpoint")
        model = LocalFinetunedModel(
            base_model=type(base),
            amago_ckpt_dir=str(Path(run_dir)),
            model_name=run_name,
            default_checkpoint=local_checkpoint,
        )
        return model, local_checkpoint, f"local:{run_name}@{local_checkpoint}"
    checkpoint = getattr(args, f"{prefix}_checkpoint")
    if checkpoint is None:
        raise ValueError(f"Missing --{prefix}-checkpoint")
    return base, checkpoint, f"SyntheticRLV2@{checkpoint}"


def run_child(args):
    prefix = args.child_role
    model, checkpoint, label = model_for(prefix, args)
    team_dir = Path(args.team_dir)
    if not team_dir.exists() or not list(team_dir.rglob("*.gen3ou_team")):
        raise FileNotFoundError(f"No Gen 3 OU teams found under {team_dir}")

    trajectory_dir = Path(args.trajectory_dir)
    db_path = Path(args.db_path)
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    teams = runner.TeamSet(str(team_dir), "gen3ou")
    print(f"[{prefix}] Loading {label} as {args.username}...", flush=True)
    agent = model.initialize_agent(checkpoint=checkpoint, log=False, action_temperature=args.temperature)
    agent.env_mode = "sync"
    agent.parallel_actors = 1
    agent.verbose = False

    logger = runner.BattleLogger(str(db_path))
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
        for key in ("model_weight", "damage_weight", "ko_weight", "switch_penalty", "anti_throw_penalty", "override_mode")
        if key in config
    })

    make_env = functools.partial(
        impl.make_local_challenge_env,
        battle_format="gen3ou",
        num_battles=args.battles,
        observation_space=model.observation_space,
        action_space=model.action_space,
        reward_function=model.reward_function,
        player_username=args.username,
        opponent_username=args.opponent_username,
        role=prefix,
        player_team_set=teams,
        save_trajectories_to=str(trajectory_dir / prefix),
        battle_ai=tactical,
        battle_logger=logger,
        decision_debug=args.decision_debug,
        transport_trace=None,
        transport_stall_seconds=60.0,
    )
    try:
        results = agent.evaluate_test([make_env], timesteps=args.battles * 1000, episodes=args.battles)
        print(f"[{prefix}] RESULTS: {results}", flush=True)
    finally:
        logger.close()
    return 0


def child_command(args, role, username, opponent, worker_dir, battles):
    role_dir = worker_dir / role
    db_path = role_dir / "battles.db"
    traj = role_dir / "trajectories"
    role_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(Path(__file__).resolve()), "--child-mode",
        "--child-role", role, "--username", username, "--opponent-username", opponent,
        "--team-dir", args.team_dir, "--battles", str(battles),
        "--temperature", str(args.temperature), "--trajectory-dir", str(traj),
        "--db-path", str(db_path),
    ]
    if args.config:
        cmd += ["--config", args.config]
    if args.decision_debug:
        cmd.append("--decision-debug")
    if getattr(args, f"{role}_type") == "checkpoint":
        cmd += [f"--{role}-checkpoint", str(getattr(args, f"{role}_checkpoint"))]
    else:
        cmd += [
            f"--{role}-run-dir", str(getattr(args, f"{role}_run_dir")),
            f"--{role}-run-name", str(getattr(args, f"{role}_run_name")),
            f"--{role}-local-checkpoint", str(getattr(args, f"{role}_local_checkpoint")),
        ]
    return cmd


def launch_worker(args, worker_id, battles, evaluation_dir):
    uid = uuid.uuid4().hex[:8]
    acceptor = f"Level5A{worker_id}{uid}"
    challenger = f"Level5C{worker_id}{uid}"
    worker_dir = evaluation_dir / "workers" / f"worker_{worker_id:02d}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    acc_cmd = child_command(args, "acceptor", acceptor, challenger, worker_dir, battles)
    cha_cmd = child_command(args, "challenger", challenger, acceptor, worker_dir, battles)
    acc_log = open(worker_dir / "acceptor.log", "w", encoding="utf-8")
    cha_log = open(worker_dir / "challenger.log", "w", encoding="utf-8")
    p_acc = subprocess.Popen(acc_cmd, cwd=REPO_ROOT, stdout=acc_log, stderr=subprocess.STDOUT, text=True)
    time.sleep(2.0)
    p_cha = subprocess.Popen(cha_cmd, cwd=REPO_ROOT, stdout=cha_log, stderr=subprocess.STDOUT, text=True)
    return {
        "id": worker_id, "battles": battles, "acceptor": acceptor, "challenger": challenger,
        "acc": p_acc, "cha": p_cha, "acc_log": acc_log, "cha_log": cha_log,
        "acc_db": worker_dir / "acceptor" / "battles.db",
        "cha_db": worker_dir / "challenger" / "battles.db",
    }


def load_finished(path, usernames):
    if not path.exists():
        raise RuntimeError(f"Missing SQLite database: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        if conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='battles'").fetchone() is None:
            raise RuntimeError(f"Missing battles table: {path}")
        rows = conn.execute("SELECT battle_id, username, opponent, winner, finished_at FROM battles").fetchall()
    except sqlite3.Error as exc:
        raise RuntimeError(f"SQLite verification failed for {path}: {exc}") from exc
    finally:
        conn.close()

    records = {}
    allowed = set(usernames)
    for row in rows:
        if not row["finished_at"]:
            continue
        username = str(row["username"] or "")
        opponent = str(row["opponent"] or "")
        if {username, opponent} != allowed:
            continue
        battle_id = str(row["battle_id"])
        winner = str(row["winner"] or "")
        if winner not in allowed:
            raise RuntimeError(f"Ambiguous winner '{winner}' in {path} for {battle_id}")
        if battle_id in records:
            raise RuntimeError(f"Duplicate battle_id '{battle_id}' in {path}")
        records[battle_id] = winner
    return records


def verify(workers):
    total = cha_wins = acc_wins = 0
    details = []
    for w in workers:
        usernames = (w["challenger"], w["acceptor"])
        acc = load_finished(w["acc_db"], usernames)
        cha = load_finished(w["cha_db"], usernames)
        if set(acc) != set(cha):
            raise RuntimeError(f"Worker {w['id']:02d} DB battle sets differ: acceptor={len(acc)} challenger={len(cha)}")
        for battle_id in acc:
            if acc[battle_id] != cha[battle_id]:
                raise RuntimeError(f"Worker {w['id']:02d} winner disagreement for {battle_id}")
            if acc[battle_id] == w["challenger"]:
                cha_wins += 1
            elif acc[battle_id] == w["acceptor"]:
                acc_wins += 1
            else:
                raise RuntimeError(f"Worker {w['id']:02d} unexpected winner for {battle_id}")
        total += len(acc)
        details.append({
            "worker": w["id"], "requested": w["battles"], "completed": len(acc),
            "challenger_wins": sum(v == w["challenger"] for v in acc.values()),
            "acceptor_wins": sum(v == w["acceptor"] for v in acc.values()),
        })
    if total != cha_wins + acc_wins:
        raise RuntimeError("Outcome invariant violated")
    return total, cha_wins, acc_wins, details


def main():
    if "--child-mode" in sys.argv:
        return run_child(parse_child())

    args = parse_parent()
    if args.battles <= 0 or args.workers <= 0:
        raise ValueError("--battles and --workers must be greater than zero")
    args.workers = min(args.workers, args.battles)
    for side in ("challenger", "acceptor"):
        if getattr(args, f"{side}_type") == "checkpoint" and getattr(args, f"{side}_checkpoint") is None:
            raise ValueError(f"Missing --{side}-checkpoint")
        if getattr(args, f"{side}_type") == "local" and any(getattr(args, f"{side}_{x}") is None for x in ("run_dir", "run_name", "local_checkpoint")):
            raise ValueError(f"Missing local arguments for {side}")

    evaluation_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    evaluation_dir = Path(args.evaluation_root) / evaluation_id
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    (evaluation_dir / "metadata.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    base, rem = divmod(args.battles, args.workers)
    workers = []
    failed = False
    error = ""
    started = time.monotonic()
    try:
        for i in range(args.workers):
            workers.append(launch_worker(args, i, base + (i < rem), evaluation_dir))
        while True:
            if all(w["acc"].poll() is not None and w["cha"].poll() is not None for w in workers):
                break
            bad = next((w for w in workers if w["acc"].poll() not in (None, 0) or w["cha"].poll() not in (None, 0)), None)
            if bad:
                failed = True
                error = f"Worker {bad['id']:02d} exited nonzero: acceptor={bad['acc'].returncode}, challenger={bad['cha'].returncode}"
                break
            if time.monotonic() - started >= args.timeout_seconds:
                failed = True
                error = f"Global timeout after {args.timeout_seconds}s"
                break
            time.sleep(1.0)
    except Exception as exc:
        failed = True
        error = str(exc)
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
                    proc.kill()
            w["acc_log"].close()
            w["cha_log"].close()

    try:
        completed, cha_wins, acc_wins, per_worker = verify(workers)
    except Exception as exc:
        completed = cha_wins = acc_wins = 0
        per_worker = []
        failed = True
        error = error or str(exc)

    complete = not failed and completed == args.battles
    artifact = {
        "evaluation_id": evaluation_id,
        "challenger": args.challenger_name,
        "challenger_model": f"SyntheticRLV2@{args.challenger_checkpoint}" if args.challenger_type == "checkpoint" else f"local:{args.challenger_run_name}@{args.challenger_local_checkpoint}",
        "acceptor": args.acceptor_name,
        "acceptor_model": f"SyntheticRLV2@{args.acceptor_checkpoint}" if args.acceptor_type == "checkpoint" else f"local:{args.acceptor_run_name}@{args.acceptor_local_checkpoint}",
        "battles_requested": args.battles,
        "battles_completed": completed,
        "challenger_wins": cha_wins,
        "acceptor_wins": acc_wins,
        "draws": 0,
        "challenger_win_rate": cha_wins / completed if completed else 0.0,
        "status": "complete" if complete else "incomplete",
        "per_worker": per_worker,
    }
    if error:
        artifact["error"] = error
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    logging.info(f"Evaluation {'VERIFIED' if complete else 'FAILED CLOSED'}: {completed}/{args.battles}")
    logging.info(f"Challenger win rate: {artifact['challenger_win_rate']:.1%}")
    return 0 if complete else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        logging.error(f"❌ {exc}")
        sys.exit(1)
