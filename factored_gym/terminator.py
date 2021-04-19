import abc

from factored_gym.factored_gym_component import FactoredGymComponent
from factored_gym.process import Process

"""
                       ______
                     <((((((\\\
                     /      . }\
                     ;--..--._|}
  (\                 '--/\--'  )
   \\                | '-'  :'|
    \\               . -==- .-|
     \\               \.__.'   \--._
     [\\          __.--|       //  _/'--.
     \ \\       .'-._ ('-----'/ __/      \
      \ \\     /   __>|      | '--.       |
       \ \\   |   \   |     /    /       /
        \ '\ /     \  |     |  _/       /
         \  \       \ |     | /        /
   snd    \  \      \        /

"""


class Terminator(FactoredGymComponent):
    
    @abc.abstractmethod
    def is_terminal(self, process: Process, observation) -> bool:
        """
        :param process: the process to check
        :param observation: agent's observation of the process
        :return: True iff the process is in a terminal state
        """
        pass
