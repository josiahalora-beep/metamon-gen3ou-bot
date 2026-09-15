"""Summarize battle_ai SQLite data.

Examples:
  python analyze_battles.py --database battle_data/battles.db --last 100
  python analyze_battles.py --database battle_data/battles.db --last 100 --json
"""
import argparse
import json
import sqlite3
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--database", default="battle_data/battles.db")
    ap.add_argument("--last", type=int, default=100)
    ap.add_argument("--json", action="store_true", dest="as_json")
    args = ap.parse_args()
    db_path = Path(args.database)
    if not db_path.exists():
        print(json.dumps({"sample_size": 0, "wins": 0, "losses": 0, "win_rate": None, "average_turns": None, "teams": {}, "our_leads": {}, "opponent_leads": {}}) if args.as_json else "No completed battles found.")
        return
    db = sqlite3.connect(db_path)
    rows = db.execute("SELECT result, team_file, our_lead, opponent_lead, final_turn FROM battles WHERE finished_at IS NOT NULL ORDER BY started_at DESC LIMIT ?", (args.last,)).fetchall()
    wins = sum(r[0] == "WIN" for r in rows)
    summary = {
        "sample_size": len(rows), "wins": wins, "losses": len(rows) - wins,
        "win_rate": (wins / len(rows)) if rows else None,
        "average_turns": (sum((r[4] or 0) for r in rows) / len(rows)) if rows else None,
        "teams": {}, "our_leads": {}, "opponent_leads": {},
    }
    for result, team, lead, opp_lead, _ in rows:
        for key, value in (("teams", team), ("our_leads", lead), ("opponent_leads", opp_lead)):
            if value not in summary[key]: summary[key][value] = {"battles": 0, "wins": 0}
            summary[key][value]["battles"] += 1
            summary[key][value]["wins"] += int(result == "WIN")
    if args.as_json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Battles: {summary['sample_size']}  Wins: {wins}  Win rate: {summary['win_rate'] if summary['win_rate'] is not None else 'N/A'}")
        print(f"Average turns: {summary['average_turns'] if summary['average_turns'] is not None else 'N/A'}")
        for section in ("teams", "our_leads", "opponent_leads"):
            print(f"\n{section}:")
            for name, item in sorted(summary[section].items(), key=lambda x: (-x[1]["battles"], str(x[0]))):
                print(f"  {name or '<unknown>'}: {item['wins']}/{item['battles']}")


if __name__ == "__main__": main()
