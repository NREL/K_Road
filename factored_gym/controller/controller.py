from abc import abstractmethod

from factored_gym.process import Process


class Controller:
    
    @abstractmethod
    def get_action(self, process: Process) -> any:
        pass
    
    def reset(self, process: 'Process'):
        pass
    
    def render(self, process: 'Process', view: 'View') -> None:
        """
        Override this to do any rendering when the gym's render() function is called
        :param process: the process being rendered
        :param view: the view to render to
        """
        pass
    
  
