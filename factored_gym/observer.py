import abc

import gym

from factored_gym.factored_gym_component import FactoredGymComponent
from factored_gym.process import Process


class Observer(FactoredGymComponent):
    
    @abc.abstractmethod
    def get_observation_space(self, process: Process) -> gym.spaces.Space:
        """
        :param process: the process being observed
        :return: an OpenAI gym compatible observation_space
        """
        # observation = self.get_observation(process)
        # return gym.spacesspaces.Box(low=-1.0, high=1.0, shape=(observation.shape[0], 1))
        pass
    
    @abc.abstractmethod
    def get_observation(self, process: Process) -> any:
        """
        :param process: the process being observed
        :return: an OpenAI gym compatible observation of the process in it's current state
        """
        pass
