#!/usr/bin/env python
import sys

from factored_gym import factored_gym
from scenario.trajectory_tracking.trajectory_tracking_observer import TrajectoryTrackingObserver
from scenario.trajectory_tracking.trajectory_tracking_process import TrajectoryTrackingProcess
from scenario.trajectory_tracking.trajectory_tracking_rewarder import TrajectoryTrackingRewarder

sys.path.append("/Users/saguasvi/cavs-environments")
import numpy as np

from scenario.trajectory_tracking.trajectory_tracking_terminator import TrajectoryTrackingTerminator


def make_target_env(
        observation_scaling=1.0,
        action_scaling=1.0 / 10.0):
    return factored_gym.FactoredGym(
        TrajectoryTrackingProcess(),
        TrajectoryTrackingObserver(observation_scaling),
        TrajectoryTrackingTerminator(),
        TrajectoryTrackingRewarder(),
        [factored_gym.ActionScaler(action_scaling)]
    )


env = make_target_env()

for i in range(100):
    obs = env.reset()
    env.render()
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
