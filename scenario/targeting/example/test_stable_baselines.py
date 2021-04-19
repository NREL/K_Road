import sys

from factored_gym import ActionScaler, FactoredGym

sys.path.append("../../../k_road/")

# import matplotlib

from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from stable_baselines.common.policies import (
    FeedForwardPolicy,
)
# from cavs_environments.vehicle.deep_road.deep_road import DeepRoad

import scenario.targeting as targeting


# import  cavs_environments.vehicle.k_road.targeting as


def make_target_env_with_baseline(
        observation_scaling=1.0,
        action_scaling=1.0 / 10.0,
        min_starting_distance: float = 50,
        max_starting_distance: float = 100,
        time_limit=60):
    return FactoredGym(
        targeting.TargetProcess(min_starting_distance, max_starting_distance),
        targeting.TargetObserver(observation_scaling, time_limit),
        targeting.TargetTerminator(time_limit),
        targeting.TargetRewarder(time_limit),
        [ActionScaler(action_scaling), targeting.TargetBaseline()]
    )


def make_target_env(
        observation_scaling=1.0,
        action_scaling=1.0,  # / 100.0,
        min_starting_distance: float = 50,
        max_starting_distance: float = 100,
        time_limit=60):
    return FactoredGym(
        targeting.TargetProcess(min_starting_distance, max_starting_distance),
        targeting.TargetObserver(observation_scaling, time_limit),
        targeting.TargetTerminator(time_limit),
        targeting.TargetRewarder(time_limit),
        [ActionScaler(action_scaling)]
    )


class CustomPolicy(FeedForwardPolicy):

    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[64, 64],
                                                          vf=[64, 64])],
                                           feature_extraction="mlp")


n_cpu = 12
env = SubprocVecEnv([lambda: make_target_env() for i in range(n_cpu)])
model = PPO2(CustomPolicy, env, verbose=1, tensorboard_log='/tmp/k_road_0/',
             gamma=.999, learning_rate=.0001)
model.learn(total_timesteps=int(5e6))
model.save('k_road_0')
print('done!')
