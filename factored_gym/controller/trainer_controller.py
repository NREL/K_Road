from ray.rllib.agents import Trainer

from factored_gym import ActionTransform
from factored_gym.controller.controller import Controller
from factored_gym.observer import Observer
from factored_gym.process import Process


class TrainerController(Controller, ActionTransform):
    
    def __init__(self, trainer: Trainer, observer: Observer):
        self._trainer: Trainer = trainer
        self._observer: Observer = observer
    
    def transform_action(self, process: Process, action):
        return self.get_action(process)
    
    def get_action(self, process: Process) -> any:
        observation = self._observer.get_observation(process)
        return self._trainer.compute_action(observation)
    
    def reset(self, process: Process) -> None:
        pass
