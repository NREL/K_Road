import math
import random

import numpy as np
from gym import spaces

from factored_gym import Observer
from k_road.constants import Constants
from k_road.util import *
from scenario.targeting import TargetProcess


class TargetObserver(Observer):

    def __init__(self, scaling: float, time_limit: float):
        self.scaling: float = scaling
        self.time_limit: float = time_limit

        self.observation_length: int = 0

        self.yaw_rate_index: int = self.observation_length
        self.observation_length += 1

        # self.steer_angle_index  : int =  self.observation_length
        # self.observation_length  : int =  self.observation_length + 1
        #
        # self.acceleration_index  : int =  self.observation_length
        # self.observation_length  : int =  self.observation_length + 1

        # self.lateral_velocity_index: int = self.observation_length
        # self.observation_length += 1

        self.longitudinal_velocity_index: int = self.observation_length
        self.observation_length += 1

        self.heading_to_target_index: int = self.observation_length
        self.observation_length += 1

        self.heading_normal_to_target_index: int = self.observation_length
        self.observation_length += 1

        self.distance_to_target_index: int = self.observation_length
        self.observation_length += 1

        self.time_left_index: int = self.observation_length
        self.observation_length += 1

        # self.baseline_acceleration_index  : int =  self.observation_length
        # self.observation_length  : int =  self.observation_length + 1
        #
        # self.baseline_steer_angle_index  : int =  self.observation_length
        # self.observation_length  : int =  self.observation_length + 1

        self.observation_space = \
            spaces.Box(low=-self.scaling, high=self.scaling, shape=(self.observation_length, 1))

    def get_observation_space(self, process: TargetProcess):
        return self.observation_space

    def get_observation(self, process: TargetProcess):
        # compute observation
        observation = np.empty(self.observation_space.shape)

        ego = process.ego_vehicle
        ego_angle = process.ego_to_target.angle
        heading = signed_delta_angle(ego.angle, ego_angle)

        observation[self.yaw_rate_index] = \
            min(1.0, max(-1.0, ego.angular_velocity / (2 * math.pi)))  # 180 deg / sec cap

        # observation[self.steer_angle_index] = ego.steer_angle / Constants.max_steer_angle
        # observation[self.acceleration_index] = ego.acceleration / Constants.max_acceleration

        # observation[self.lateral_velocity_index] = \
        #     min(1.0, max(-1.0, ego.lateral_velocity / 11.176))  # 25 mph cap

        # observation[self.longitudinal_velocity_index] = \
        #     min(1, max(-1, ego.longitudinal_velocity / Constants.max_speed))
        # print(ego.longitudinal_velocity, Constants.max_speed)
        observation[self.longitudinal_velocity_index] = \
            min(1.0, max(-1.0, (ego.longitudinal_velocity / Constants.max_speed)))

        observation[self.heading_to_target_index] = math.sin(heading)
        observation[self.heading_normal_to_target_index] = math.cos(heading)

        observation[self.distance_to_target_index] = \
            min(1.0, max(-1.0, 1 - (process.distance_to_target / process.max_distance_from_target)))

        observation[self.time_left_index] = 1 - (process.time / self.time_limit)

        # baseline_steer_rate, baseline_jerk = self.get_baseline_action()
        # # observation[self.baseline_acceleration_index] = baseline_jerk / Constants.max_jerk
        # # observation[self.baseline_steer_angle_index] = baseline_steer_rate / Constants.max_steer_rate
        # observation[self.baseline_acceleration_index] = \
        #     inverse_two_sided_exponential(2, baseline_jerk / Constants.max_jerk)
        # observation[self.baseline_steer_angle_index] = \
        #     inverse_two_sided_exponential(2, baseline_steer_rate / Constants.max_steer_rate)
        # # observation[self.baseline_acceleration_index] = baseline_acceleration / Constants.max_acceleration
        # # observation[self.baseline_steer_angle_index] = max_baseline_steer_angle / Constants.max_steer_angle

        observation = np.multiply(self.scaling, observation)

        if random.random() < 1e-2:
            print(np.transpose(observation))
        return observation
