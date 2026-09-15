"""
Play a Metamon pretrained agent against RANDOM opponents on the real,
public Pokemon Showdown ladder (sim3.psim.us) - not the PokeAgent
research ladder, not your local server.
"""

import argparse
import functools

from poke_env.ps_client.server_configuration import ServerConfiguration

from metamon.env.wrappers import QueueOnLocalLadder
from metamon.rl.metamon_to_amago import PSLadderAMAGOWrapper, _block_warnings
from metamon.rl.pretrained import SmallRL
from metamon.env.wrappers import get_metamon_teams


PublicShowdownServerConfiguration = ServerConfiguration(
    "wss://sim3.psim.us/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)


class PublicLadder(QueueOnLocalLadder):
    _INIT_RETRIES = 1000

    @property
    def server_configuration(self):
        return PublicShowdownServerConfiguration


def make_public_ladder_env(*args, **kwargs):
    _block_warnings()
    menv = PublicLadder(*args, **kwargs)
    print("Made Public Ladder Env")
    return PSLadderAMAGOWrapper(menv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", default=None)
    parser.add_argument("--battle_format", default="gen1ou")
    parser.add_argument("--battles", type=int, default=5)
    parser.add_argument("--team_set", default="competitive")
    args = parser.parse_args()

    pretrained_model = SmallRL()

    if "random" in args.battle_format.lower():
        team_set = None
    else:
        team_set = get_metamon_teams(args.battle_format, args.team_set)

    agent = pretrained_model.initialize_agent(checkpoint=None, log=False, action_temperature=1.0)
    agent.env_mode = "sync"
    agent.parallel_actors = 1
    agent.verbose = False

    make_env = functools.partial(
        make_public_ladder_env,
        observation_space=pretrained_model.observation_space,
        action_space=pretrained_model.action_space,
        reward_function=pretrained_model.reward_function,
        num_battles=args.battles,
        team_preview_model=None,
        player_username=args.username,
        player_password=args.password,
        player_team_set=team_set,
        battle_backend=pretrained_model.battle_backend,
        battle_format=args.battle_format,
    )

    print(f"Connecting as '{args.username}' to the PUBLIC Showdown ladder ({args.battle_format})...")
    results = agent.evaluate_test([make_env], timesteps=args.battles * 1000, episodes=args.battles)
    print(results)


if __name__ == "__main__":
    main()
