import asyncio

from metamon.public_ladder_hardening import install_public_ladder_hardening
from poke_env.ps_client.ps_client import PSClient


class _FakeClient:
    def __init__(self):
        self.messages = []

    async def set_team(self, packed_team):
        self.messages.append(("team", packed_team))

    async def send_message(self, message, room="", message_2=None):
        self.messages.append((message, room, message_2))


def test_public_ladder_search_disables_hidden_and_invite_only():
    install_public_ladder_hardening()
    fake = _FakeClient()

    asyncio.run(
        PSClient.search_ladder_game(fake, "gen3ou", "packed-team")
    )

    assert fake.messages == [
        ("team", "packed-team"),
        ("/hidenext off", "", None),
        ("/inviteonlynext off", "", None),
        ("/search gen3ou", "", None),
    ]


def test_public_ladder_hardening_is_idempotent():
    install_public_ladder_hardening()
    install_public_ladder_hardening()
    assert getattr(PSClient, "search_ladder_game") is not None
