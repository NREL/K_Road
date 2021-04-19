from typing import Optional

import numpy as np

from factored_gym.controller.controller import Controller
from factored_gym.process import Process


class NullController(Controller):
    
    def __init__(self, action_space):
        self._action_space_shape = action_space.shape
    
    def get_action(self, process: Optional[Process]) -> any:
        return np.zeros(self._action_space_shape)
    
    def compute_action(self, observation, *args, **kwargs):
        return self.get_action(None)
