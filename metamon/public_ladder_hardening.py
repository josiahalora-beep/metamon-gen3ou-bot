"""Public Showdown ladder hardening for Metamon's poke-env client.

This is intentionally installed at Metamon import time so the public ladder
wrapper does not need to depend on a particular poke-env source checkout.
"""

from __future__ import annotations

import json
import requests

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
        # Preserve Showdown's normal privacy behavior. The authenticated
        # account may intentionally use hidden/invite-only battles; the bot
        # should not force /hidenext or /inviteonlynext on or off.
        return await original_search_ladder_game(self, format_, packed_team)

    async def public_log_in(self, split_message):
        """Authenticate against the current Showdown login API.

        Metamon pins an older poke-env release whose login implementation sends
        the legacy ``act=login`` field. Current Showdown routes the operation
        through ``/api/login`` itself, so the request body must contain only
        the username, password, and challenge string fields needed by that API.
        """
        if self.account_configuration.password:
            challstr = split_message[2] + "%7C" + split_message[3]
            response = requests.post(
                self.server_configuration.authentication_url,
                data={
                    "name": self.account_configuration.username,
                    "pass": self.account_configuration.password,
                    "challstr": challstr,
                },
                timeout=10.0,
            )
            self.logger.info("Sending authentication request")
            try:
                data = json.loads(response.text[1:])
            except (ValueError, IndexError) as exc:
                self.logger.error(
                    "Showdown login returned a non-JSON response (HTTP %s)",
                    response.status_code,
                )
                raise RuntimeError("Showdown authentication response was invalid") from exc

            assertion = data.get("assertion")
            if not assertion:
                error = data.get("error") or data.get("actionerror") or "unknown authentication error"
                self.logger.error(
                    "Showdown authentication failed: %s (HTTP %s)",
                    error,
                    response.status_code,
                )
                raise RuntimeError(f"Showdown authentication failed: {error}")
        else:
            self.logger.info("Bypassing authentication request")
            assertion = ""

        await self.send_message(f"/trn {self.username},0,{assertion}")
        await self.change_avatar(self._avatar)

    async def resilient_handle_message(message: str):
        try:
            # original_handle_message is the unbound class method captured
            # above, so preserve the instance explicitly.
            return await original_handle_message(self, message)
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
    PSClient.log_in = public_log_in
    PSClient._handle_message = resilient_handle_message
    _INSTALLED = True
