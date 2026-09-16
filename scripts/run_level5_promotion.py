#!/usr/bin/env python3
"""
Authoritative Level 5 promotion controller for Gen 3 OU local self-play.

Promotion is transactional:
  1. Validate candidate + champion manifest.
  2. Evaluate candidate vs immutable anchor.
  3. Evaluate candidate vs current champion.
  4. Evaluate candidate vs a deterministic rotating historical pool.
  5. Validate every evaluation artifact and aggregate exact battle counts.
  6. Write a promotion decision artifact.
  7. Only on success, atomically replace champion_manifest.json.

The controller fails closed. A crashed evaluator, incomplete database, malformed
result, duplicate battle count, missing checkpoint, or ambiguous metric cannot
promote a candidate.

This script intentionally uses the authoritative run_level5_evaluation.py for
battle execution. It owns policy/orchestration only; it does not attempt to
reconstruct battle results from logs itself.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


LOG = logging.getLogger("level5-promotion")
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
EVALUATOR = SCRIPT_DIR / "run_level5_evaluation.py"

ANCHOR_MIN_WR = 0.45
CHAMPION_MIN_WR = 0.52
HISTORICAL_MIN_WR = 0.50
OVERALL_MIN_WR = 0.55
DEFAULT_HISTORICAL_COUNT = 2
SCHEMA_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed Level 5 Gen 3 OU champion promotion controller"
    )

    # Candidate identity/model.
    parser.add_argument("--candidate-type", choices=["checkpoint", "local"], required=True)
    parser.add_argument("--candidate-checkpoint")
    parser.add_argument("--candidate-run-dir")
    parser.add_argument("--candidate-run-name")
    parser.add_argument("--candidate-local-checkpoint")
    parser.add_argument("--candidate-id", required=True)

    # Promotion state/artifacts.
    parser.add_argument(
        "--manifest-path",
        default="promotions/champion_manifest.json",
        help="Champion state manifest",
    )
    parser.add_argument(
        "--evaluation-root",
        default="promotions/history",
        help="Directory in which immutable promotion evidence is archived",
    )
    parser.add_argument("--team-dir", required=True)

    # Evaluation policy.
    parser.add_argument("--battles-per-tier", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--historical-count", type=int, default=DEFAULT_HISTORICAL_COUNT)
    parser.add_argument(
        "--allow-empty-history",
        action="store_true",
        help="Allow promotion with no historical opponents. Disabled by default (fail closed).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and show the planned tiers without running battles or changing state",
    )
    return parser.parse_args()


def die(message: str, exit_code: int = 1) -> None:
    LOG.error("REJECTED: %s", message)
    raise SystemExit(exit_code)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_component(value: str, fallback: str = "candidate") -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return value[:96] or fallback


def canonical_json(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        die(f"Required JSON file not found: {path}")
    except json.JSONDecodeError as exc:
        die(f"Malformed JSON at {path}: {exc}")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def validate_manifest(manifest: Any, manifest_path: Path) -> None:
    if not isinstance(manifest, dict):
        die(f"Manifest must be a JSON object: {manifest_path}")

    if manifest.get("schema_version") not in (None, 1, SCHEMA_VERSION):
        die(f"Unsupported champion manifest schema_version={manifest.get('schema_version')!r}")

    for key in ("anchor", "champion"):
        if not isinstance(manifest.get(key), dict):
            die(f"Manifest is missing required object: {key}")
        validate_model_spec(manifest[key], f"manifest.{key}")

    pool = manifest.get("historical_pool", [])
    if not isinstance(pool, list):
        die("manifest.historical_pool must be a list")

    ids = set()
    for index, entry in enumerate(pool):
        if not isinstance(entry, dict):
            die(f"historical_pool[{index}] must be an object")
        validate_model_spec(entry, f"manifest.historical_pool[{index}]")
        entry_id = entry["id"]
        if entry_id in ids:
            die(f"Duplicate historical_pool id: {entry_id}")
        ids.add(entry_id)


def validate_model_spec(spec: Dict[str, Any], label: str) -> None:
    if not isinstance(spec.get("id"), str) or not spec["id"].strip():
        die(f"{label}.id must be a non-empty string")
    model_type = spec.get("type", "checkpoint")
    if model_type not in ("checkpoint", "local"):
        die(f"{label}.type must be checkpoint or local")

    if model_type == "checkpoint":
        path = spec.get("path")
        if not isinstance(path, str) or not path:
            die(f"{label}.path is required for checkpoint models")
        if not Path(path).expanduser().is_file():
            die(f"{label}.path does not exist: {path}")
    else:
        for key in ("run_dir", "run_name", "local_checkpoint"):
            if not isinstance(spec.get(key), str) or not spec[key]:
                die(f"{label}.{key} is required for local models")
        run_dir = Path(spec["run_dir"]).expanduser()
        if not run_dir.is_dir():
            die(f"{label}.run_dir does not exist: {run_dir}")


def validate_candidate(args: argparse.Namespace) -> None:
    if args.candidate_type == "checkpoint":
        if not args.candidate_checkpoint:
            die("--candidate-checkpoint is required for checkpoint candidates")
        path = Path(args.candidate_checkpoint).expanduser()
        if not path.is_file():
            die(f"Candidate checkpoint does not exist: {path}")
    else:
        required = {
            "--candidate-run-dir": args.candidate_run_dir,
            "--candidate-run-name": args.candidate_run_name,
            "--candidate-local-checkpoint": args.candidate_local_checkpoint,
        }
        for flag, value in required.items():
            if not value:
                die(f"{flag} is required for local candidates")
        if not Path(args.candidate_run_dir).expanduser().is_dir():
            die(f"Candidate run directory does not exist: {args.candidate_run_dir}")

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", args.candidate_id):
        die("--candidate-id must contain only A-Z/a-z/0-9/._- and be <= 96 characters")


def build_candidate_args(args: argparse.Namespace) -> List[str]:
    if args.candidate_type == "checkpoint":
        return [
            "--challenger-type", "checkpoint",
            "--challenger-checkpoint", args.candidate_checkpoint,
        ]
    return [
        "--challenger-type", "local",
        "--challenger-run-dir", args.candidate_run_dir,
        "--challenger-run-name", args.candidate_run_name,
        "--challenger-local-checkpoint", args.candidate_local_checkpoint,
    ]


def build_opponent_args(spec: Dict[str, Any]) -> List[str]:
    model_type = spec.get("type", "checkpoint")
    if model_type == "checkpoint":
        return [
            "--acceptor-type", "checkpoint",
            "--acceptor-checkpoint", spec["path"],
            "--acceptor-name", spec["id"],
        ]
    return [
        "--acceptor-type", "local",
        "--acceptor-run-dir", spec["run_dir"],
        "--acceptor-run-name", spec["run_name"],
        "--acceptor-local-checkpoint", spec["local_checkpoint"],
        "--acceptor-name", spec["id"],
    ]


def validate_result(result: Any, tier_name: str, expected_battles: int) -> Dict[str, Any]:
    if not isinstance(result, dict):
        die(f"{tier_name}: evaluator result is not an object")
    if result.get("status") != "complete":
        die(f"{tier_name}: evaluator status is not complete: {result.get('status')!r}")

    integer_fields = ("battles_completed", "challenger_wins", "acceptor_wins")
    for field in integer_fields:
        value = result.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            die(f"{tier_name}: invalid integer field {field}={value!r}")

    battles = result["battles_completed"]
    cw = result["challenger_wins"]
    aw = result["acceptor_wins"]
    if battles != expected_battles:
        die(f"{tier_name}: expected exactly {expected_battles} verified battles, got {battles}")
    if cw + aw != battles:
        die(f"{tier_name}: wins do not reconcile: challenger={cw}, acceptor={aw}, battles={battles}")

    wr = result.get("challenger_win_rate")
    if not isinstance(wr, (int, float)) or isinstance(wr, bool) or not 0.0 <= float(wr) <= 1.0:
        die(f"{tier_name}: invalid challenger_win_rate={wr!r}")
    calculated = cw / battles if battles else 0.0
    if abs(float(wr) - calculated) > 1e-9:
        die(f"{tier_name}: reported win rate does not equal wins/battles")

    return result


def run_tier_evaluation(
    tier_name: str,
    promotion_dir: Path,
    candidate_args: List[str],
    opponent_spec: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    tier_dir = promotion_dir / safe_component(tier_name)
    tier_dir.mkdir(parents=True, exist_ok=True)
    output_json = tier_dir / "result.json"
    evaluator_root = tier_dir / "evaluator_workspace"
    stdout_path = tier_dir / "stdout.log"
    stderr_path = tier_dir / "stderr.log"

    cmd = [
        sys.executable,
        str(EVALUATOR),
        *candidate_args,
        *build_opponent_args(opponent_spec),
        "--battles", str(args.battles_per_tier),
        "--workers", str(args.workers),
        "--team-dir", args.team_dir,
        "--evaluation-root", str(evaluator_root),
        "--temperature", str(args.temperature),
        "--timeout-seconds", str(args.timeout_seconds),
        "--output-json", str(output_json),
        "--challenger-name", args.candidate_id,
    ]

    LOG.info("EVALUATING %-24s | %d battles vs %s", tier_name, args.battles_per_tier, opponent_spec["id"])
    started = utc_now()
    try:
        with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
            completed = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=out,
                stderr=err,
                text=True,
                timeout=max(1, args.timeout_seconds + 120),
                check=False,
            )
    except subprocess.TimeoutExpired:
        die(f"{tier_name}: evaluator process exceeded controller timeout")

    if completed.returncode != 0:
        die(f"{tier_name}: evaluator exited with code {completed.returncode}; see {stderr_path}")
    if not output_json.is_file():
        die(f"{tier_name}: evaluator succeeded but produced no result.json")

    result = validate_result(load_json(output_json), tier_name, args.battles_per_tier)
    result["controller_started_at"] = started
    result["controller_verified_at"] = utc_now()
    result["result_sha256"] = sha256_file(output_json)
    save_json(output_json, result)
    return result


def historical_key(spec: Dict[str, Any]) -> str:
    return str(spec.get("id"))


def select_historical_opponents(
    manifest: Dict[str, Any],
    count: int,
    champion_id: str,
) -> List[Dict[str, Any]]:
    pool = manifest.get("historical_pool", [])
    candidates = [x for x in pool if x.get("id") != champion_id]
    if not candidates:
        return []

    # Deterministic rotation that advances with manifest iteration while avoiding
    # duplicates. This keeps coverage broad without evaluating every champion.
    iteration = int(manifest.get("iteration", 0) or 0)
    start = iteration % len(candidates)
    ordered = candidates[start:] + candidates[:start]
    return ordered[:count]


def result_summary(tier: str, result: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    return {
        "tier": tier,
        "opponent_id": result.get("acceptor_name") or result.get("opponent") or "unknown",
        "battles_completed": result["battles_completed"],
        "challenger_wins": result["challenger_wins"],
        "acceptor_wins": result["acceptor_wins"],
        "challenger_win_rate": float(result["challenger_win_rate"]),
        "threshold": threshold,
        "passed": float(result["challenger_win_rate"]) >= threshold,
        "result_sha256": result.get("result_sha256"),
    }


def build_new_champion_spec(args: argparse.Namespace) -> Dict[str, Any]:
    if args.candidate_type == "checkpoint":
        return {
            "id": args.candidate_id,
            "type": "checkpoint",
            "path": str(Path(args.candidate_checkpoint).expanduser()),
        }
    return {
        "id": args.candidate_id,
        "type": "local",
        "run_dir": str(Path(args.candidate_run_dir).expanduser()),
        "run_name": args.candidate_run_name,
        "local_checkpoint": args.candidate_local_checkpoint,
    }


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Parse what is actually on disk before replacing live state.
        load_json(tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def archive_previous_manifest(manifest_path: Path, promotion_dir: Path) -> Optional[Path]:
    if not manifest_path.is_file():
        return None
    archive = promotion_dir / "previous_champion_manifest.json"
    shutil.copy2(manifest_path, archive)
    return archive


def write_rejection_artifact(
    promotion_dir: Path,
    eval_id: str,
    candidate_id: str,
    reason: str,
    results: Dict[str, Any],
) -> None:
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_id": eval_id,
        "candidate_id": candidate_id,
        "status": "rejected",
        "reason": reason,
        "timestamp": utc_now(),
        "tiers": results,
        "manifest_modified": False,
    }
    atomic_write_json(promotion_dir / "promotion_result.json", artifact)


def main() -> None:
    args = parse_args()

    if args.battles_per_tier < 1:
        die("--battles-per-tier must be >= 1")
    if args.workers < 1:
        die("--workers must be >= 1")
    if args.timeout_seconds < 1:
        die("--timeout-seconds must be >= 1")
    if args.historical_count < 0:
        die("--historical-count must be >= 0")
    if not 0.0 < args.temperature:
        die("--temperature must be > 0")
    if not EVALUATOR.is_file():
        die(f"Authoritative evaluator missing: {EVALUATOR}")
    if not Path(args.team_dir).expanduser().is_dir():
        die(f"Team directory does not exist: {args.team_dir}")

    manifest_path = Path(args.manifest_path).expanduser()
    manifest = load_json(manifest_path)
    validate_manifest(manifest, manifest_path)
    validate_candidate(args)

    champion = manifest["champion"]
    anchor = manifest["anchor"]
    if args.candidate_id == champion["id"]:
        die("Candidate ID is already the current champion; refusing self-promotion")

    historical = select_historical_opponents(manifest, args.historical_count, champion["id"])
    if not historical and not args.allow_empty_history:
        die(
            "No historical opponents are available. Add historical_pool entries or explicitly pass "
            "--allow-empty-history."
        )

    tiers: List[Tuple[str, Dict[str, Any], float]] = [
        ("anchor", anchor, ANCHOR_MIN_WR),
        ("champion", champion, CHAMPION_MIN_WR),
    ]
    for index, opponent in enumerate(historical):
        tiers.append((f"historical_{index:02d}_{safe_component(opponent['id'])}", opponent, HISTORICAL_MIN_WR))

    required_battles = len(tiers) * args.battles_per_tier
    eval_id = f"promotion_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')}_{safe_component(args.candidate_id)}"
    promotion_dir = Path(args.evaluation_root).expanduser() / eval_id
    promotion_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_id": eval_id,
        "candidate_id": args.candidate_id,
        "candidate_type": args.candidate_type,
        "created_at": utc_now(),
        "repo_root": str(REPO_ROOT),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "policy": {
            "anchor_min_win_rate": ANCHOR_MIN_WR,
            "champion_min_win_rate": CHAMPION_MIN_WR,
            "historical_min_win_rate": HISTORICAL_MIN_WR,
            "overall_min_win_rate": OVERALL_MIN_WR,
            "battles_per_tier": args.battles_per_tier,
            "historical_count": len(historical),
            "required_verified_battles": required_battles,
        },
        "tiers": [
            {"name": name, "opponent_id": spec["id"], "threshold": threshold}
            for name, spec, threshold in tiers
        ],
    }
    atomic_write_json(promotion_dir / "metadata.json", metadata)

    LOG.info("=" * 72)
    LOG.info("LEVEL 5 PROMOTION CONTROLLER")
    LOG.info("Candidate: %s", args.candidate_id)
    LOG.info("Promotion ID: %s", eval_id)
    LOG.info("Tiers: %d | Required verified battles: %d", len(tiers), required_battles)
    LOG.info("Anchor: >= %.1f%% | Champion: >= %.1f%% | Historical: >= %.1f%% | Overall: >= %.1f%%",
             ANCHOR_MIN_WR * 100, CHAMPION_MIN_WR * 100, HISTORICAL_MIN_WR * 100, OVERALL_MIN_WR * 100)
    LOG.info("Artifacts: %s", promotion_dir)

    if args.dry_run:
        LOG.info("DRY RUN: no battles executed and champion manifest was not modified.")
        return

    candidate_args = build_candidate_args(args)
    verified: Dict[str, Dict[str, Any]] = {}

    try:
        for tier_name, opponent, threshold in tiers:
            result = run_tier_evaluation(tier_name, promotion_dir, candidate_args, opponent, args)
            verified[tier_name] = result
            wr = float(result["challenger_win_rate"])
            LOG.info("  %s: %.1f%% (%d/%d)", tier_name, wr * 100, result["challenger_wins"], result["battles_completed"])
            if wr < threshold:
                reason = f"{tier_name} gate failed: {wr:.4f} < {threshold:.4f}"
                write_rejection_artifact(promotion_dir, eval_id, args.candidate_id, reason, verified)
                die(f"{reason}. Champion manifest remains unchanged.")
    except SystemExit:
        raise
    except Exception as exc:
        write_rejection_artifact(
            promotion_dir,
            eval_id,
            args.candidate_id,
            f"controller exception: {type(exc).__name__}: {exc}",
            verified,
        )
        die(f"Unexpected controller failure: {type(exc).__name__}: {exc}")

    total_battles = sum(x["battles_completed"] for x in verified.values())
    total_wins = sum(x["challenger_wins"] for x in verified.values())
    if total_battles != required_battles:
        write_rejection_artifact(
            promotion_dir, eval_id, args.candidate_id,
            f"verified battle floor failed: {total_battles} != {required_battles}", verified,
        )
        die(f"Verified battle floor failed: {total_battles} != {required_battles}")

    overall_wr = total_wins / total_battles
    if overall_wr < OVERALL_MIN_WR:
        reason = f"overall gate failed: {overall_wr:.4f} < {OVERALL_MIN_WR:.4f}"
        write_rejection_artifact(promotion_dir, eval_id, args.candidate_id, reason, verified)
        die(f"{reason}. Champion manifest remains unchanged.")

    # Persist the complete, independently verified decision before touching champion state.
    promotion_artifact = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_id": eval_id,
        "candidate_id": args.candidate_id,
        "status": "promoted",
        "timestamp": utc_now(),
        "manifest_modified": False,
        "required_verified_battles": required_battles,
        "verified_battles": total_battles,
        "overall_win_rate": overall_wr,
        "gates": {
            "anchor": result_summary("anchor", verified["anchor"], ANCHOR_MIN_WR),
            "champion": result_summary("champion", verified["champion"], CHAMPION_MIN_WR),
            "historical": [
                result_summary(name, verified[name], HISTORICAL_MIN_WR)
                for name, _, _ in tiers
                if name.startswith("historical_")
            ],
            "overall": {
                "wins": total_wins,
                "battles": total_battles,
                "win_rate": overall_wr,
                "threshold": OVERALL_MIN_WR,
                "passed": overall_wr >= OVERALL_MIN_WR,
            },
        },
    }
    atomic_write_json(promotion_dir / "promotion_result.json", promotion_artifact)

    # Build next manifest from a deep copy so a failed serialization/validation can never
    # mutate the in-memory state used as the source of truth.
    old_manifest = copy.deepcopy(manifest)
    new_manifest = copy.deepcopy(manifest)
    old_champion = copy.deepcopy(new_manifest["champion"])
    new_champion = build_new_champion_spec(args)

    pool = list(new_manifest.get("historical_pool", []))
    if not any(historical_key(x) == old_champion["id"] for x in pool):
        pool.append(old_champion)
    new_manifest["historical_pool"] = pool
    new_manifest["champion"] = new_champion
    new_manifest["iteration"] = int(new_manifest.get("iteration", 0) or 0) + 1
    new_manifest["schema_version"] = SCHEMA_VERSION
    new_manifest["last_promotion"] = {
        "evaluation_id": eval_id,
        "candidate_id": args.candidate_id,
        "overall_win_rate": overall_wr,
        "anchor_win_rate": float(verified["anchor"]["challenger_win_rate"]),
        "champion_win_rate": float(verified["champion"]["challenger_win_rate"]),
        "verified_battles": total_battles,
        "timestamp": utc_now(),
    }

    history_ids = list(new_manifest.get("history", []))
    if old_champion["id"] not in history_ids:
        history_ids.insert(0, old_champion["id"])
    new_manifest["history"] = history_ids

    # Validate the exact manifest that is about to replace live state.
    validate_manifest(new_manifest, manifest_path)

    archive = archive_previous_manifest(manifest_path, promotion_dir)
    atomic_write_json(manifest_path, new_manifest)

    # Verify the committed manifest and only then mark the artifact as state-changing.
    committed = load_json(manifest_path)
    validate_manifest(committed, manifest_path)
    promotion_artifact["manifest_modified"] = True
    promotion_artifact["manifest_sha256_before"] = sha256_file(archive) if archive else None
    promotion_artifact["manifest_sha256_after"] = sha256_file(manifest_path)
    promotion_artifact["new_champion"] = new_champion
    promotion_artifact["previous_champion"] = old_champion
    atomic_write_json(promotion_dir / "promotion_result.json", promotion_artifact)

    LOG.info("=" * 72)
    LOG.info("PROMOTED: %s -> Champion", args.candidate_id)
    LOG.info("Overall: %.1f%% across %d verified battles", overall_wr * 100, total_battles)
    LOG.info("Champion manifest updated atomically: %s", manifest_path)
    LOG.info("Promotion evidence: %s", promotion_dir)


if __name__ == "__main__":
    main()
