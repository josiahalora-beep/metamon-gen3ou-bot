from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class BattleLogger:
    """Crash-tolerant SQLite logger. Credentials are never accepted or stored."""
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS battles (
          battle_id TEXT PRIMARY KEY, started_at TEXT, finished_at TEXT,
          username TEXT, opponent TEXT, format TEXT, team_file TEXT,
          our_lead TEXT, opponent_lead TEXT, winner TEXT, result TEXT,
          final_turn INTEGER, rating REAL, gxe REAL, replay_url TEXT, loss_class TEXT
        );
        CREATE TABLE IF NOT EXISTS turns (
          id INTEGER PRIMARY KEY AUTOINCREMENT, battle_id TEXT, turn INTEGER,
          snapshot_json TEXT NOT NULL, candidate_actions_json TEXT,
          chosen_action INTEGER, final_action INTEGER, reasoning_json TEXT,
          created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS inferences (
          id INTEGER PRIMARY KEY AUTOINCREMENT, battle_id TEXT, pokemon TEXT,
          data_json TEXT, created_at TEXT
        );
        """)
        self.db.commit()

    def _now(self): return datetime.now(timezone.utc).isoformat()

    def start(self, battle_id, *, username="", opponent="", format="gen3ou", team_file=""):
        with self.lock:
            self.db.execute("INSERT OR IGNORE INTO battles(battle_id,started_at,username,opponent,format,team_file) VALUES(?,?,?,?,?,?)",
                            (battle_id, self._now(), username, opponent, format, team_file))
            self.db.commit()

    def turn(self, battle_id, turn, snapshot, candidates, chosen, final, reasoning):
        with self.lock:
            self.db.execute("INSERT INTO turns(battle_id,turn,snapshot_json,candidate_actions_json,chosen_action,final_action,reasoning_json,created_at) VALUES(?,?,?,?,?,?,?,?)",
                            (battle_id, turn, json.dumps(snapshot, default=str), json.dumps(candidates, default=str), int(chosen), int(final), json.dumps(reasoning, default=str), self._now()))
            self.db.commit()

    def finish(self, battle_id, *, opponent_lead="", winner="", result="", final_turn=0,
               rating=None, gxe=None, replay_url="", loss_class=""):
        with self.lock:
            self.db.execute("UPDATE battles SET finished_at=?,opponent_lead=?,winner=?,result=?,final_turn=?,rating=?,gxe=?,replay_url=?,loss_class=? WHERE battle_id=?",
                            (self._now(), opponent_lead, winner, result, final_turn, rating, gxe, replay_url, loss_class, battle_id))
            self.db.commit()

    def close(self): self.db.close()
