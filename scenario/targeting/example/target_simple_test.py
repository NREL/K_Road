# import sys
#
# sys.path.append("../../../")

import numpy as np

# import  cavs_environments.vehicle.k_road.targeting as
from factored_gym import ActionScaler, FactoredGym
# import matplotlib
# from cavs_environments.vehicle.deep_road.deep_road import DeepRoad
from scenario.targeting import TargetProcess, TargetObserver, TargetTerminator, TargetRewarder, TargetBaseline


def make_target_env_with_baseline(
        observation_scaling=1.0,
        action_scaling=1.0 / 10.0,
        max_distance_from_target=125,
        time_limit=60):
    return FactoredGym(
        TargetProcess(max_distance_from_target),
        TargetObserver(time_limit, observation_scaling),
        TargetTerminator(time_limit),
        TargetRewarder(time_limit),
        [ActionScaler(action_scaling), TargetBaseline()]
    )


# def make_target_env(
#         observation_scaling=1.0,
#         action_scaling=1.0 / 10.0,
#         max_distance_from_target=125,
#         time_limit=60):
#     return framework.FactoredGym(
#         targeting.TargetProcess(max_distance_from_target),
#         targeting.TargetObserver(time_limit, observation_scaling),
#         targeting.TargetTerminator(time_limit),
#         targeting.TargetRewarder(),
#         [framework.ActionScaler(action_scaling)]
#     )


env = make_target_env_with_baseline()
env.reset()
env.render()

obs = None
for _ in range(1000):
    action = np.empty(2)
    action[0] = 0  # np.random.normal(.5, .001)
    action[1] = 0  # np.random.normal(.5, .001)

    result = env.step(action)
    obs = result[0]
    env.render()
    if result[2]:
        break
env.close()
