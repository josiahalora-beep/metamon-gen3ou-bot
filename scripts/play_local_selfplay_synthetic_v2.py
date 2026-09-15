"""Compatibility launcher for local SyntheticRLV2 self-play.

The self-play environment uses the existing BattleSessionState safety checks. A
ChallengeByUsername reset begins a new challenge rather than entering the public
ladder queue, so the session must explicitly enter QUEUING before reset.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import play_local_selfplay_synthetic as impl


_ORIGINAL_RESET = impl.LocalSyntheticChallenge.reset
_ORIGINAL_CHILD_COMMAND = impl.child_command


def _patched_reset(self, *args, **kwargs):
    # ChallengeByUsername.reset() is the point where the challenge is requested.
    # Mirror the public runner's lifecycle contract before it creates the battle.
    if self._session.phase == impl.runner.SessionPhase.ENDED:
        self._session.cleanup_started()
        self._session.cleanup_finished()
    if self._session.phase == impl.runner.SessionPhase.CLEANUP:
        self._session.cleanup_finished()
    if self._session.phase == impl.runner.SessionPhase.IDLE:
        self._session.begin_queue()
    return _ORIGINAL_RESET(self, *args, **kwargs)


def _patched_child_command(args, role, username, opponent_username, log_dir):
    cmd = _ORIGINAL_CHILD_COMMAND(args, role, username, opponent_username, log_dir)
    cmd[1] = str(Path(__file__).resolve())
    return cmd


impl.LocalSyntheticChallenge.reset = _patched_reset
impl.child_command = _patched_child_command


if __name__ == "__main__":
    asyncio.run(impl.main())
