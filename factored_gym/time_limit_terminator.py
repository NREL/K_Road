import abc

from factored_gym import (
    ProcessTimed,
    Terminator,
    )
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


class TimeLimitTerminator(Terminator):
    """
    Terminates after reaching a given time limit.
    """
    
    def __init__(self, time_limit: float):
        self.time_limit: float = time_limit
    
    def is_terminal(self, process: ProcessTimed, observation) -> bool:
        """
        :param process: the process to check
        :param observation: agent's observation of the process
        :return: True iff the process is in a terminal state
        """
        return process.time >= self.time_limit
