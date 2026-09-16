from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n))) / d
    return max(0.0, c - h), min(1.0, c + h)


def discover_dbs(paths: list[str]) -> list[Path]:
    found: set[Path] = set()
    for raw in paths:
        p = Path(raw)
        if p.is_file() and p.name.endswith(".db"):
            found.add(p.resolve())
        elif p.is_dir():
            found.update(x.resolve() for x in p.rglob("*.db") if x.is_file())
    return sorted(found)


def read_rows(path: Path) -> list[dict]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        table = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='battles'").fetchone()
        if table is None:
            return []
        cols = {r[1] for r in conn.execute("PRAGMA table_info(battles)")}
        wanted = [c for c in ("battle_id", "username", "opponent", "result", "winner", "team_file", "our_lead", "opponent_lead", "final_turn", "finished_at", "started_at") if c in cols]
        if not wanted:
            return []
        order = " ORDER BY started_at DESC" if "started_at" in cols else ""
        return [dict(r) for r in conn.execute(f"SELECT {','.join(wanted)} FROM battles WHERE finished_at IS NOT NULL{order}").fetchall()]
    finally:
        conn.close()


def add_bucket(bucket: dict, key: str | None, won: bool) -> None:
    name = key or "<unknown>"
    item = bucket.setdefault(name, {"battles": 0, "wins": 0})
    item["battles"] += 1
    item["wins"] += int(won)


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate Gen 3 OU SQLite battle logs into auditable statistics.")
    ap.add_argument("--database", action="append", default=[], help="SQLite DB or directory; may be repeated.")
    ap.add_argument("--root", action="append", default=[], help="Directory to recursively scan for battles.db files.")
    ap.add_argument("--output", default="analysis_exports/battle_analysis.json")
    ap.add_argument("--elo-k", type=float, default=32.0)
    args = ap.parse_args()

    inputs = args.database + args.root
    if not inputs:
        inputs = ["battle_data"]
    dbs = discover_dbs(inputs)

    players: dict[str, float] = defaultdict(lambda: 1500.0)
    pair: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"battles": 0, "wins": 0})
    teams: dict = {}
    our_leads: dict = {}
    opp_leads: dict = {}
    db_reports = []
    total = wins = losses = draws = 0
    turns = []

    for db in dbs:
        rows = read_rows(db)
        db_wins = db_losses = db_draws = 0
        for r in rows:
            result = str(r.get("result") or "").upper()
            username = str(r.get("username") or "").strip()
            opponent = str(r.get("opponent") or "").strip()
            won = result == "WIN" or (r.get("winner") and str(r["winner"]) == username)
            lost = result == "LOSS" or (r.get("winner") and str(r["winner"]) == opponent)
            if not won and not lost:
                draws += 1; db_draws += 1
                outcome = 0.5
            elif won:
                wins += 1; db_wins += 1; outcome = 1.0
            else:
                losses += 1; db_losses += 1; outcome = 0.0
            total += 1
            if r.get("final_turn") is not None:
                try: turns.append(float(r["final_turn"]))
                except (TypeError, ValueError): pass
            if username and opponent and username != opponent:
                a, b = sorted((username, opponent))
                key = (a, b)
                pair[key]["battles"] += 1
                pair[key]["wins"] += int((username == a and outcome == 1.0) or (username == b and outcome == 0.0))
                ra, rb = players[a], players[b]
                expected_a = 1 / (1 + 10 ** ((rb - ra) / 400))
                score_a = outcome if username == a else 1 - outcome
                players[a] += args.elo_k * (score_a - expected_a)
                players[b] += args.elo_k * ((1 - score_a) - (1 - expected_a))
            add_bucket(teams, r.get("team_file"), bool(won))
            add_bucket(our_leads, r.get("our_lead"), bool(won))
            add_bucket(opp_leads, r.get("opponent_lead"), bool(won))
        db_reports.append({"database": str(db), "completed_battles": len(rows), "wins": db_wins, "losses": db_losses, "draws_or_unknown": db_draws})

    lo, hi = wilson(wins, total)
    result = {
        "schema_version": 1,
        "scope": {"databases": [str(x) for x in dbs]},
        "summary": {"battles": total, "wins": wins, "losses": losses, "draws_or_unknown": draws, "win_rate": (wins / total if total else None), "win_rate_wilson_95": [lo, hi], "average_final_turn": (sum(turns) / len(turns) if turns else None)},
        "databases": db_reports,
        "elo": sorted(({"player": p, "rating": round(r, 2)} for p, r in players.items()), key=lambda x: -x["rating"]),
        "matchups": [{"player_a": a, "player_b": b, **v, "win_rate_a": v["wins"] / v["battles"] if v["battles"] else None} for (a, b), v in sorted(pair.items())],
        "teams": teams,
        "our_leads": our_leads,
        "opponent_leads": opp_leads,
        "data_capabilities": {"battle_outcomes": True, "elo": True, "team_and_lead_usage": True, "trajectory_state_wpa": False, "damage_roll_model": False, "policy_kl": False},
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Databases analyzed: {len(dbs)}")
    print(f"Report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
