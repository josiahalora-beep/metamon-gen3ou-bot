from __future__ import annotations

import sqlite3
from pathlib import Path


class LeadDatabase:
    """Historical lead statistics; it never assumes team preview exists in ADV."""
    def __init__(self, path: str | Path):
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS leads(team TEXT, our_lead TEXT, opponent_lead TEXT, result TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
        self.db.commit()

    def record(self, team, our_lead, opponent_lead, result):
        self.db.execute("INSERT INTO leads(team,our_lead,opponent_lead,result) VALUES(?,?,?,?)", (team, our_lead, opponent_lead, result))
        self.db.commit()

    def matchup_rate(self, team, our_lead, opponent_lead):
        row = self.db.execute("SELECT COUNT(*), SUM(result='WIN') FROM leads WHERE team=? AND our_lead=? AND opponent_lead=?", (team, our_lead, opponent_lead)).fetchone()
        return (float(row[1] or 0) / row[0]) if row and row[0] else None

    def close(self): self.db.close()
