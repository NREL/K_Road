import abc

from factored_gym.process import Process


class ProcessTimed(Process):
    """
    An abstract class representing a process that keeps track of a simulated time
    """
    
    @property
    @abc.abstractmethod
    def time(self) -> float:
        """
        :return: the current process time
        """
        pass
    
    @property
    @abc.abstractmethod
    def time_step_length(self) -> float:
        """
        :return: the change in time between the last and the current time step
        """
        pass
