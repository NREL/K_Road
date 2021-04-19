import math

from factored_gym import ActionTransform
from k_road.util import *
from scenario.targeting import TargetProcess


class TargetBaseline(ActionTransform):

    def __init__(self, max_baseline_steer_angle=45 / math.pi):
        self.max_baseline_steer_angle = max_baseline_steer_angle

    def transform_action(self, process: TargetProcess, action):
        ego = process.ego_vehicle
        delta = process.ego_to_target
        # delta_distance = delta.get_length()
        heading = signed_delta_angle(ego.angle, delta.angle)

        baseline_acceleration = 1 if ego.speed < 3 else 0
        # baseline_jerk = 2 * (baseline_acceleration - ego.acceleration)
        acceleration = baseline_acceleration + action[0]

        baseline_steer_angle = clamp(-.3 * heading, -self.max_baseline_steer_angle, self.max_baseline_steer_angle)
        # baseline_steer_rate = 5 * (baseline_steer_angle - ego.steer_angle)
        steer_angle = baseline_steer_angle + action[1]

        return acceleration, steer_angle

        #
        # baseline_steer_rate, baseline_jerk = self.get_baseline_action()
        # # baseline_steer_rate, baseline_jerk = (0, 0)
        #
        # jerk_raw_action = action[1] / self.action_scaling
        # # jerk_raw_action = 0
        # jerk_action_mod = two_sided_exponential(2, jerk_raw_action)
        # jerk_action = baseline_jerk + jerk_action_mod * Constants.max_jerk
        #
        # steer_raw_action = action[0] / self.action_scaling
        # # steer_raw_action = 0
        # steer_action_mod = two_sided_exponential(1.5, steer_raw_action)
        # steer_action = max(-Constants.max_steer_rate,
        #                    min(Constants.max_steer_rate,
        #                        baseline_steer_rate + steer_action_mod * (
        #                                self.max_baseline_steer_angle + Constants.max_steer_rate)))
        # # steer_action = steer_action_mod
        # # steer_action = max(-Constants.max_steer_angle,
        # #                    min(Constants.max_steer_angle,
        # #                        baseline_steer_angle + steer_action_mod * Constants.max_steer_angle))
        #
        # # self.ego_vehicle.apply_control(self.time_step, jerk_action, steer_action)
