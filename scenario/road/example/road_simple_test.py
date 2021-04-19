# import sys
#
# sys.path.append("../../../")

import numpy as np

import scenario.road as road


# import matplotlib
# from cavs_environments.vehicle.deep_road.deep_road import DeepRoad


# import  cavs_environments.vehicle.k_road.targeting as

def make_target_env_with_baseline(
        observation_scaling=1.0,  # 10.0
        ego_starting_distance=150.0
):
    return factored_gym.FactoredGym(
        road.RoadProcess(ego_starting_distance=ego_starting_distance),
        road.RoadObserver(observation_scaling),
        road.RoadTerminator(time_limit=3 * 60),
        road.RoadGoalRewarder(),
        []
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
        env.reset()
env.close()
