"""Run the existing SyntheticRLV2 ladder runner against a local Showdown server.

The runner module keeps all battle AI, logging, and checkpoint behavior in one place.
We only replace its public server configuration at import time so the exact same
policy/evaluator path can be stress-tested locally without touching live ladder code.
"""

from __future__ import annotations

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


LOCAL_SERVER = ServerConfiguration(
    "ws://127.0.0.1:8000/showdown/websocket",
    "http://127.0.0.1:8000/action.php?",
)


async def main() -> None:
    runner.PUBLIC_SERVER = LOCAL_SERVER
    await runner.main()


if __name__ == "__main__":
    asyncio.run(main())
