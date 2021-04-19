from factored_gym import (
    ActionTransform,
    Controller,
    )
from factored_gym.process import Process


class AdditiveActionTransform(ActionTransform):
    
    def __init__(self, base_coefficient: float, input_coefficient: float, input: Controller):
        self.input: Controller = input
        self.base_coefficient: float = base_coefficient
        self.input_coefficient: float = input_coefficient
    
    def transform_action(self, process: Process, action):
        # print('atf: {} {}'.format(action, self.input.get_action(process)))
        return action * self.base_coefficient + self.input.get_action(process) * self.input_coefficient
    
    def reset(self, process: 'Process') -> None:
        self.input.reset(process)
    
    # def close(self, process: 'Process') -> None:
    #     self._input.close(process)
    #
    # def begin_render(self, process: 'Process', view: 'View') -> None:
    #     self._input.begin_render(process, view)
    
    def render(self, process: 'Process', view: 'View') -> None:
        self.input.render(process, view)
