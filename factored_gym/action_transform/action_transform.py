import abc

from factored_gym.factored_gym_component import FactoredGymComponent
from factored_gym.process import Process


class ActionTransform(FactoredGymComponent):
    
    def get_action_space(self, process: Process, source_action_space):
        """
        Override this to specify a different action space
        :param process: the process being observed
        :param source_action_space: the action space of the source (the source is the process if this is the first
        transform)
        :return: an OpenAI gym compatible action_space
        """
        return source_action_space
    
    @abc.abstractmethod
    def transform_action(self, process: Process, action):
        """
        Transforms the given action.
        :param process: the process being observed
        :param action: an untransformed action
        :return: a transformed action, possibly ready for use in the process's step() function
        """
        pass
