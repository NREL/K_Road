import math

from factored_gym.rewarder import Rewarder


class Discounter(Rewarder):
    
    def __init__(self, discount_rate, delegate_rewarder):
        self.discount_rate = discount_rate
        self.delegate_rewarder = delegate_rewarder
    
    def get_reward(self, process, observation, terminated):
        return self.delegate_rewarder.get_reward(process, observation, terminated) * \
               math.pow(self.discount_rate, process.get_time())
