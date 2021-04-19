from factored_gym.action_transform.action_transform import ActionTransform
from factored_gym.process import Process


class LinearActionCenterer(ActionTransform):
    
    def __init__(self, gains: [float], offsets: [float]):
        self.gains: [float] = gains
        self.offsets: [float] = offsets
    
    def transform_action(self, process: Process, action):
        for i in range(action.size):
            action[i] = LinearActionCenterer.linear_offset(self.offsets[i], action[i])
        return action
    
    @staticmethod
    def linear_offset(offset, x):
        if x < offset:
            return (x + 1) / (offset + 1) - 1
        else:
            return (x - offset) / (1 - offset)
