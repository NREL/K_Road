import abc

from factored_gym.factored_gym_component import FactoredGymComponent
from factored_gym.process import Process


class Rewarder(FactoredGymComponent):
    
    @abc.abstractmethod
    def get_reward(self, process: Process, observation, terminated: bool) -> float:
        """
        :param process: the process to compute a reward from
        :param observation: the agent's observation of the process
        :param terminated: True iff the process is in a terminal state
        :return: numerical reward for reaching this process state
        """
        pass
