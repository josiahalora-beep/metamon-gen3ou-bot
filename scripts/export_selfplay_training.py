"""Export local self-play SQLite battles into a compact AMAGO-style trajectory dataset.

The exporter deliberately keeps the original JSON snapshots and actions rather than
pretending that they are a different observation encoding.  It creates JSONL records
with one record per turn plus battle-level outcome metadata, suitable for a downstream
training adapter.  Incomplete battles are rejected by default.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def load_db(path: Path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    battles = {r["battle_id"] if "battle_id" in r.keys() else str(r["id"]): dict(r)
               for r in conn.execute("SELECT rowid AS battle_id, * FROM battles")}
    # Most versions key turns by the battle id stored in battle_id. Also support
    # string ids if the schema has an explicit id column in future revisions.
    turns = list(conn.execute("SELECT * FROM turns ORDER BY battle_id, turn, id"))
    conn.close()
    return battles, turns


def export_one(db_path: Path, out, source: str, require_finished: bool) -> tuple[int, int]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    battle_rows = conn.execute("SELECT rowid AS battle_rowid, * FROM battles").fetchall()
    battles = {}
    for b in battle_rows:
        d = dict(b)
        battles[str(d["battle_rowid"])] = d
        # Accommodate schemas where battle_id is stored as a textual key.
        battles.setdefault(str(d.get("battle_id", "")), d)

    count_battles = 0
    count_turns = 0
    for b in battle_rows:
        b = dict(b)
        if require_finished and not b.get("finished_at"):
            continue
        battle_key = str(b["battle_rowid"])
        rows = conn.execute(
            "SELECT * FROM turns WHERE battle_id = ? ORDER BY turn, id", (battle_key,)
        ).fetchall()
        if not rows:
            # Some builds use the textual battle key instead of rowid.
            rows = conn.execute(
                "SELECT * FROM turns WHERE battle_id = ? ORDER BY turn, id",
                (str(b.get("battle_id", "")),),
            ).fetchall()
        if not rows:
            continue

        count_battles += 1
        for r in rows:
            r = dict(r)
            try:
                snapshot = json.loads(r["snapshot_json"])
            except Exception:
                snapshot = r["snapshot_json"]
            try:
                candidates = json.loads(r["candidate_actions_json"]) if r.get("candidate_actions_json") else None
            except Exception:
                candidates = r.get("candidate_actions_json")
            try:
                reasoning = json.loads(r["reasoning_json"]) if r.get("reasoning_json") else None
            except Exception:
                reasoning = r.get("reasoning_json")

            record = {
                "source": source,
                "battle": {
                    "id": battle_key,
                    "username": b.get("username"),
                    "opponent": b.get("opponent"),
                    "format": b.get("format"),
                    "team_file": b.get("team_file"),
                    "our_lead": b.get("our_lead"),
                    "opponent_lead": b.get("opponent_lead"),
                    "winner": b.get("winner"),
                    "result": b.get("result"),
                    "final_turn": b.get("final_turn"),
                    "loss_class": b.get("loss_class"),
                },
                "turn": r.get("turn"),
                "observation": snapshot,
                "candidate_actions": candidates,
                "chosen_action": r.get("chosen_action"),
                "final_action": r.get("final_action"),
                "reasoning": reasoning,
            }
            out.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")
            count_turns += 1
    conn.close()
    return count_battles, count_turns


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="local_fast directory")
    p.add_argument("--output", required=True, help="JSONL output path")
    p.add_argument("--allow-incomplete", action="store_true")
    args = p.parse_args()

    root = Path(args.input)
    dbs = [
        (root / "selfplay_acceptor" / "battles.db", "acceptor"),
        (root / "selfplay_challenger" / "battles.db", "challenger"),
    ]
    missing = [str(p) for p, _ in dbs if not p.exists()]
    if missing:
        raise SystemExit("Missing self-play databases: " + ", ".join(missing))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    total_battles = total_turns = 0
    with out.open("w", encoding="utf-8") as f:
        for db, source in dbs:
            b, t = export_one(db, f, source, not args.allow_incomplete)
            print(f"{source}: {b} battles, {t} turns exported")
            total_battles += b
            total_turns += t
    print(f"TOTAL: {total_battles} battles, {total_turns} turns")
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
