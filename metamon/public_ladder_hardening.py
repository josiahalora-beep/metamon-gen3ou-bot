"""Public Showdown ladder hardening for Metamon's poke-env client.

This is intentionally installed at Metamon import time so the public ladder
wrapper does not need to depend on a particular poke-env source checkout.
"""

from __future__ import annotations

from poke_env.ps_client.ps_client import PSClient


_INSTALLED = False


def install_public_ladder_hardening() -> None:
    """Install idempotent fixes needed for public Showdown laddering."""
    global _INSTALLED
    if _INSTALLED:
        return

    original_search_ladder_game = PSClient.search_ladder_game
    original_handle_message = PSClient._handle_message

    async def public_search_ladder_game(self, format_: str, packed_team: str | None):
        # Showdown's current privacy controls are /hidenext and
        # /inviteonlynext. Older /ionext is intentionally obsolete.
        await self.set_team(packed_team)
        await self.send_message("/hidenext off")
        await self.send_message("/inviteonlynext off")
        await self.send_message(f"/search {format_}")

    async def resilient_handle_message(self, message: str):
        try:
            return await original_handle_message(message)
        except ValueError:
            text = str(message or "").lower()
            # Privacy/battle-room notices should never tear down the message
            # handling task. The actual battle stream continues on subsequent
            # websocket messages.
            privacy_notice = (
                "invite-only" in text
                or "hidden battle" in text
                or "hidden battles" in text
                or "eavesdropper" in text
            )
            if privacy_notice:
                self.logger.warning(
                    "Ignored malformed Showdown privacy notice: %s", message
                )
                return None
            raise

    PSClient.search_ladder_game = public_search_ladder_game
    PSClient._handle_message = resilient_handle_message
    _INSTALLED = True
