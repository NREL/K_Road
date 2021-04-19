import abc

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from factored_gym import Process
    from factored_gym import View


class FactoredGymComponent(abc.ABC):
    
    def reset(self, process: 'Process') -> None:
        """
        Called to reset the process to an initial state.
        :param process: the process being reset
        """
        pass
    
    def close(self, process: 'Process') -> None:
        """
        Closes the gym.
        :param process: the process being closed
        """
        pass
    
    def begin_render(self, process: 'Process', view: 'View') -> None:
        """
        Override this to do any rendering when the gym's render() function is called.
        begin_render() is called on each component before render() is called.
        :param process: the process being rendered
        :param view: the view to render to
        """
        pass
    
    def render(self, process: 'Process', view: 'View') -> None:
        """
        Override this to do any rendering when the gym's render() function is called
        :param process: the process being rendered
        :param view: the view to render to
        """
        pass
