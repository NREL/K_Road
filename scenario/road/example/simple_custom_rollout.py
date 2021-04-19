#!/usr/bin/env python

import argparse
import os
import pickle

import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from scenario import FactoredModel
from scenario import PathPlanningModel

# from ray.rllib.utils.space_utils import flatten_to_single_ndarray

EXAMPLE_USAGE = """
Example Usage via RLlib CLI:
    rllib rollout /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl

Example Usage via executable:
    ./rollout.py /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl
"""

ENV = "PathPlanningEnv"
if ENV == "CarlaRoadEnv":
    def env_creator(env_config):
        env = CarlaRoadEnv(env_config)
        return env


    register_env("CarlaRoadEnv-v0", env_creator)
    ModelCatalog.register_custom_model("carla_road_model", FactoredModel)
else:
    def env_creator(env_config):
        env = PathPlanningEnv(env_config)
        return env


    register_env("PathPlanningEnv-v0", env_creator)
    ModelCatalog.register_custom_model("path_planning_model", PathPlanningModel)


# Note: if you use any custom models or envs, register them here first, e.g.:
#
# ModelCatalog.register_custom_model("pa_model", ParametricActionsModel)
# register_env("pa_cartpole", lambda _: ParametricActionCartpole(10))


def create_parser(parser_creator=None):
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Roll out a reinforcement learning agent "
                    "given a checkpoint.",
        epilog=EXAMPLE_USAGE)

    parser.add_argument(
        "checkpoint", type=str, help="Checkpoint from which to roll out.")
    parser.add_argument(
        "--steps",
        default=10000,
        help="Number of timesteps to roll out (overwritten by --episodes).")
    parser.add_argument(
        "--episodes",
        default=0,
        help="Number of complete episodes to roll out (overrides --steps).")

    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Specifies KRoad On Carla Map scenario (one of: cross_intersection, curvy_road, avoid_bots)")
    parser.add_argument(
        "--with-carla",
        default=False,
        action="store_true",
        help="Run CARLA in puppet mode alongside k_road rollouts")

    return parser


def run(args, parser):
    config = {}
    # Load configuration from checkpoint file.
    config_dir = os.path.dirname(args.checkpoint)
    print("loading from ", config_dir)

    config_path = os.path.join(config_dir, "params.pkl")
    # Try parent directory.
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../../params.pkl")

    print("loading from ", config_path)
    # If no pkl file found, require command line `--config`.
    if not os.path.exists(config_path):
        if not args.config:
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory AND no config given on command line!")

    # Load the config from pickled.
    else:
        with open(config_path, "rb") as f:
            config = pickle.load(f)

    from test_standard_carla_map_factored_obs import CarlaRoadEnv
    from train_path_planning import PathPlanningEnv
    if args.scenario is not None:
        config['env_config']['scenario'] = args.scenario
    config['env_config']['with_carla'] = args.with_carla

    config['num_workers'] = 0
    if ENV == "CarlaRoadEnv":
        env = CarlaRoadEnv(config['env_config'])  ## note, actual creat the env object; passed to rollout.
        agent = ppo.PPOTrainer(config=config, env="CarlaRoadEnv-v0")
    else:
        env = PathPlanningEnv(config['env_config'])  ## note, actual creat the env object; passed to rollout.
        agent = ppo.PPOTrainer(config=config, env="PathPlanningEnv-v0")
    print('restore trainer')
    agent.restore(args.checkpoint)

    num_steps = int(args.steps)
    num_episodes = int(args.episodes)

    rollout(agent, env, num_steps, num_episodes)


def keep_going(steps, num_steps, episodes, num_episodes):
    """Determine whether we've collected enough data"""
    # if num_episodes is set, this overrides num_steps
    if num_episodes:
        return episodes < num_episodes
    # if num_steps is set, continue until we reach the limit
    if num_steps:
        return steps < num_steps
    # otherwise keep going forever
    return True


def rollout(agent,
            env,
            num_steps,
            num_episodes=0,
            no_render=False):
    steps = 0
    episodes = 0
    while keep_going(steps, num_steps, episodes, num_episodes):
        reward_total = 0
        done = False
        print("BEGINNING AN EPISODE")
        obs = env.reset()
        while not done and keep_going(steps, num_steps, episodes, num_episodes):
            action = agent.compute_action(obs)
            next_obs, reward, done, info = env.step(action)
            reward_total += reward
            if not no_render:
                env.render()
            steps += 1
            obs = next_obs
        print("Episode #{}: reward: {}".format(episodes, reward_total))
        if True or reward_total < 0:
            import time
            time.sleep(1)
        if done:
            episodes += 1


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
