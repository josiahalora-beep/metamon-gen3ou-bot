"""Run the existing SyntheticRLV2 ladder runner against a local Showdown server.

The runner module keeps all battle AI, logging, and checkpoint behavior in one place.
We only replace its public server configuration at import time so the exact same
policy/evaluator path can be stress-tested locally without touching live ladder code.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# This script is executed from scripts/, so Python otherwise cannot import the
# repository-root module play_public_synthetic.py.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import play_public_synthetic as runner
from poke_env.ps_client.server_configuration import ServerConfiguration


# Local Showdown uses the same Smogon authentication endpoint as poke-env's
# LocalhostServerConfiguration. The local --no-security server does not need a
# real password, so normalize our PowerShell-safe sentinel to None before the
# shared runner builds AccountConfiguration.
LOCAL_PASSWORD_SENTINEL = "local"
_original_parse_args = argparse.ArgumentParser.parse_args


def _parse_local_args(self, *args, **kwargs):
    parsed = _original_parse_args(self, *args, **kwargs)
    if getattr(parsed, "password", None) == LOCAL_PASSWORD_SENTINEL:
        parsed.password = None
    return parsed


argparse.ArgumentParser.parse_args = _parse_local_args


LOCAL_SERVER = ServerConfiguration(
    "ws://127.0.0.1:8000/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)


async def main() -> None:
    runner.PUBLIC_SERVER = LOCAL_SERVER
    await runner.main()


if __name__ == "__main__":
    asyncio.run(main())
