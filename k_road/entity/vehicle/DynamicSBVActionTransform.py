import gym
import numpy as np

from factored_gym import ActionTransform
from factored_gym.process import Process
from k_road.constants import Constants
from k_road.util import (
    clamp,
    hinge_transform,
)


class DynamicSBVActionTransform(ActionTransform):

    def __init__(self):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=[2], dtype=np.float32)

    def get_action_space(self, process: Process, source_action_space):
        return self.action_space

    def transform_action(self, process: Process, action):
        acceleration = hinge_transform(clamp(action[0]), Constants.max_deceleration, Constants.max_acceleration)
        steer_angle = clamp(action[1]) * Constants.max_steer_angle
        # print(action, np.array([acceleration, steer_angle]))
        return np.array([acceleration, steer_angle])
