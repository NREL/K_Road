import gym
import numpy as np

from factored_gym.action_transform.action_transform import ActionTransform
from factored_gym.process import Process


class ActionIntegrator(ActionTransform):
    
    def __init__(self):
        self.action = None
    
    def reset(self, process: Process):
        self.action = None
    
    def get_action_space(self, process: Process, source_action_space):
        return gym.spaces.Box(
            low=np.multiply(1.0 / self.gain, source_action_space.low),
            high=np.multiply(1.0 / self.gain, source_action_space.high),
            dtype=source_action_space.dtype)
    
    def transform_action(self, process: Process, action):
        if self.action is None:
            self.action = action
        else:
            self.action = np.add(self.action, action)
        return self.action
