from factored_gym import Terminator
from scenario.road import RoadProcess


class RoadTerminator(Terminator):

    def __init__(self, time_limit: float = 3 * 60):
        self.time_limit: float = time_limit

    def is_terminal(self, process: RoadProcess, observation) -> bool:
        # reached_target = process.distance_to_target <= 0
        # reached_target = False
        reached_target = process.ego_in_end_zone
        exceeded_time_limit = process.time >= self.time_limit
        collided = process.ego_vehicle.collided
        terminal = reached_target or exceeded_time_limit or collided
        # if terminal:
        #     print('terminal: ', reached_target, collided, exceeded_time_limit)
        return terminal
