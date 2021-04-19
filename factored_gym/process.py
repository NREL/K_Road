import abc
from typing import Any

import gym

from factored_gym.factored_gym_component import FactoredGymComponent
from factored_gym.view import View


class Process(FactoredGymComponent):
    """
    An abstract class representing a process used in a FactoredGym
    """
    
    @abc.abstractmethod
    def get_action_space(self) -> gym.spaces.Space:
        """
        :return: an OpenAI gym compatible action_space
        """
        pass
    
    @abc.abstractmethod
    def step(self, action) -> Any:
        """
        Advances the process one timestep while applying the given action.
        :param action: action to take at this timestep
        :return any extra info you want to return from the step() call
        """
        pass
    
    @abc.abstractmethod
    def make_view(self, mode) -> View:
        """
        Factory method which returns a View to render the process
        :param mode: OpenAI Gym mode parameter
        :return: the View
        """
        pass
