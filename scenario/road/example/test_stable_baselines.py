import sys

import stable_baselines

sys.path.append("../../../k_road/")

import numpy as np

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import (
    OrnsteinUhlenbeckActionNoise,
)

from stable_baselines import DDPG
# from cavs_environments.vehicle.deep_road.deep_road import DeepRoad

import scenario.road as road


# def make_target_env_with_baseline(
#     observation_scaling = 1.0,
#     action_scaling = 1.0 / 10.0,
#     max_distance_from_target = 125,
#     time_limit = 60):

#     return framework.FactoredGym(
#         targeting.TargetProcess(time_limit, max_distance_from_target),
#         targeting.TargetObserver(observation_scaling),
#         targeting.TargetTerminator(),
#         targeting.TargetRewarder(),
#         [framework.ActionScaler(action_scaling), targeting.TargetBaseline()]
#         )

class ThisRoadEnv(factored_gym.FactoredGym):

    def __init__(self, env_config):
        observation_scaling = 1.0  # 10.0
        ego_starting_distance = 200.0
        super().__init__(
            road.RoadProcess(ego_starting_distance=ego_starting_distance),
            road.RoadObserver(observation_scaling),
            road.RoadTerminator(time_limit=5 * 60),
            road.RoadGoalRewarder(),
            # [framework.ActionScaler(1.0/10.0), framework.ActionCenterer([.001, 5], [0, 0])]
            [factored_gym.ActionCenterer([10, 10], [0, 0])]
        )


class CustomPolicy(stable_baselines.ddpg.policies.FeedForwardPolicy):

    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args,
                                           layers=[128, 128, 128, 128],
                                           layer_norm=True,
                                           feature_extraction="mlp",
                                           **kwargs
                                           )


# class CustomPolicy(MlpPolicy):
#     def __init__(self, *args, **kwargs):
#         super(MlpPolicy, self).__init__(*args, act_fun=tf.nn.tanh, net_arch=[32, 32])

# register_policy('LargeMLP', LargeMLP)

env = DummyVecEnv([lambda: ThisRoadEnv(None)])

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model = DDPG(CustomPolicy, env, verbose=1, tensorboard_log='/tmp/k_road_0/',
             gamma=.999, param_noise=param_noise, action_noise=action_noise)

model.learn(total_timesteps=int(100e3))
model.save('k_road_test')
print('done!')
