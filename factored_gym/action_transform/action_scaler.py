import gym
import numpy as np

from factored_gym.action_transform.action_transform import ActionTransform
from factored_gym.process import Process


class ActionScaler(ActionTransform):
    
    def __init__(self, gain: float):
        self.gain = gain
    
    def get_action_space(self, process: Process, source_action_space):
        return gym.spaces.Box(
            low=np.multiply(1.0 / self.gain, source_action_space.low),
            high=np.multiply(1.0 / self.gain, source_action_space.high),
            dtype=source_action_space.dtype)
    
    def transform_action(self, process: Process, action):
        return np.multiply(self.gain, action)
