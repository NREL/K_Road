from factored_gym import Terminator
from scenario.targeting import TargetProcess


class TargetTerminator(Terminator):

    def __init__(self, time_limit: float):
        self.time_limit: float = time_limit

    def is_terminal(self, process: TargetProcess, observation) -> bool:
        reached_target = process.distance_to_target <= 0
        exceeded_time_limit = process.time >= self.time_limit
        out_of_bounds = process.distance_to_target > process.max_distance_from_target

        return reached_target or exceeded_time_limit or out_of_bounds
