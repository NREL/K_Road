#!/usr/bin/env python3
import os
import sys

from scenario.trajectory_tracking.controller.pid_controllers import VehiclePIDController
from scenario.trajectory_tracking.trajectory_tracking_observer import TrajectoryTrackingObserver
from scenario.trajectory_tracking.trajectory_tracking_process import TrajectoryTrackingProcess
from scenario.trajectory_tracking.trajectory_tracking_rewarder import TrajectoryTrackingRewarder

sys.path.append(str(os.environ["HOME"]) + "/cavs-environments")

from factored_gym import ActionScaler, FactoredGym
from scenario.trajectory_tracking.trajectory_tracking_terminator import TrajectoryTrackingTerminator

"""
    Tuning taken from : https://www.hindawi.com/journals/ijvt/2014/259465/
"""


def make_target_env(
        observation_scaling=1.0,
        action_scaling=1.0):
    return FactoredGym(
        TrajectoryTrackingProcess(time_dilation=1.),
        TrajectoryTrackingObserver(observation_scaling),  # where is this
        TrajectoryTrackingTerminator(max_cross_track_error=25),
        TrajectoryTrackingRewarder(),
        [VehiclePIDController(args_lateral= \
                                  {'K_P': 1., 'K_D': -5.0, 'K_I': 1.0}, \
                              args_longitudinal= \
                                  {'K_P': 0.0, 'K_D': 0.0, 'K_I': 0.0}, \
                              look_ahead_multiplier=20.), \
         ActionScaler(action_scaling)]
    )


env = make_target_env()

for i in range(100):
    obs = env.reset()
    env.render()
    while True:
        action = np.empty(2)
        action[0] = 0
        action[1] = 0
        result = env.step(action)
        obs = result[0]
        env.render()
        if result[2]:
            break
env.close()
