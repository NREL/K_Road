import math

from factored_gym.action_transform.action_transform import ActionTransform
from factored_gym.process import Process


class ActionCenterer(ActionTransform):
    
    def __init__(self, gains: [float], offsets: [float]):
        self.gains: [float] = gains
        self.offsets: [float] = offsets
    
    def transform_action(self, process: Process, action):
        for i in range(action.size):
            action[i] = ActionCenterer.two_sided_offset_exponential(self.gains[i], self.offsets[i], action[i])
        return action
    
    @staticmethod
    def two_sided_offset_exponential(gain, offset, x):
        y = 0
        s = 1
        if x < offset:
            y = 1 - (x + 1) / (offset + 1)
            s = -1
        else:
            y = (x - offset) / (1 - offset)
        return s * (math.exp(gain * y) - 1) / (math.exp(gain) - 1)
