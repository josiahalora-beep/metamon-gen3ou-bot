"""
Public PokÃ©mon Showdown Gen 3 OU diagnostic.

Purpose:
    1. Connect to the REAL public PokÃ©mon Showdown server.
    2. Log in with the supplied account.
    3. Submit YOUR custom Gen 3 OU team.
    4. Search the public Gen 3 OU ladder.
    5. Play the requested number of battles using poke-env's RandomPlayer.

This is ONLY a connection/team diagnostic.
It does NOT load Metamon or SyntheticRLV2.

Usage:
    python diagnose_public_ladder.py ^
        --username "YOUR_SHOWDOWN_USERNAME" ^
        --password "YOUR_SHOWDOWN_PASSWORD" ^
        --team_file gen3_rain.txt ^
        --battles 1
"""

import argparse
import asyncio
from pathlib import Path

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.ps_client.server_configuration import ServerConfiguration


PublicShowdownServerConfiguration = ServerConfiguration(
    "wss://sim3.psim.us/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)


DEFAULT_TEAM = """Kingdra @ Leftovers
Ability: Swift Swim
EVs: 252 SpA / 4 SpD / 252 Spe
Modest Nature
- Rain Dance
- Surf
- Ice Beam
- Hidden Power [Electric]

Ludicolo @ Leftovers
Ability: Swift Swim
EVs: 252 SpA / 4 SpD / 252 Spe
Modest Nature
- Surf
- Giga Drain
- Leech Seed
- Substitute

Omastar @ Leftovers
Ability: Swift Swim
EVs: 252 SpA / 4 SpD / 252 Spe
Modest Nature
- Rain Dance
- Surf
- Ice Beam
- Spikes

Zapdos @ Leftovers
Ability: Pressure
EVs: 252 HP / 4 SpA / 252 Spe
Timid Nature
- Thunder
- Hidden Power [Grass]
- Rest
- Roar

Swampert @ Leftovers
Ability: Torrent
EVs: 240 HP / 216 Def / 52 SpA
Relaxed Nature
- Earthquake
- Hydro Pump
- Ice Beam
- Roar

Metagross @ Leftovers
Ability: Clear Body
EVs: 252 HP / 176 Atk / 80 Def
Adamant Nature
- Meteor Mash
- Earthquake
- Explosion
- Protect
"""


def load_team(team_file: str | None) -> str:
    """Load a Showdown-format team from disk or use the built-in rain team."""
    if team_file is None:
        return DEFAULT_TEAM

    path = Path(team_file)

    if not path.exists():
        raise FileNotFoundError(f"Team file not found: {path}")

    team = path.read_text(encoding="utf-8").strip()

    if not team:
        raise ValueError(f"Team file is empty: {path}")

    return team


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test a custom team on the real public PokÃ©mon Showdown Gen 3 OU ladder."
    )

    parser.add_argument(
        "--username",
        required=True,
        help="Your PokÃ©mon Showdown username.",
    )

    parser.add_argument(
        "--password",
        default=None,
        help="Your PokÃ©mon Showdown password.",
    )

    parser.add_argument(
        "--battle_format",
        default="gen3ou",
        help="Showdown format. Default: gen3ou",
    )

    parser.add_argument(
        "--team_file",
        default=None,
        help="Path to a Showdown-exported team text file.",
    )

    parser.add_argument(
        "--battles",
        type=int,
        default=1,
        help="Number of ladder battles.",
    )

    args = parser.parse_args()

    team = load_team(args.team_file)

    print()
    print("========================================")
    print(" Public Showdown Ladder Diagnostic")
    print("========================================")
    print(f"Username: {args.username}")
    print(f"Format:   {args.battle_format}")
    print(f"Battles:  {args.battles}")
    print("Team:     Custom Gen 3 rain team")
    print()

    account_config = AccountConfiguration(
        args.username,
        args.password,
    )

    player = RandomPlayer(
        account_configuration=account_config,
        server_configuration=PublicShowdownServerConfiguration,
        battle_format=args.battle_format,
        team=team,
    )

    print(
        f"Connecting as '{args.username}' "
        f"and searching for {args.battles} "
        f"{args.battle_format} battle(s)..."
    )

    try:
        await player.ladder(args.battles)
    finally:
        # Prevent lingering websocket/client tasks from keeping Python alive.
        await player.ps_client.stop()

    print()
    print("========================================")
    print(" Finished")
    print("========================================")
    print(f"Battles finished: {player.n_finished_battles}")
    print(f"Battles won:      {player.n_won_battles}")

    if player.n_finished_battles:
        win_rate = (
            player.n_won_battles / player.n_finished_battles
        ) * 100.0
        print(f"Win rate:         {win_rate:.1f}%")
    else:
        print("Win rate:         N/A")


if __name__ == "__main__":
    asyncio.run(main())
