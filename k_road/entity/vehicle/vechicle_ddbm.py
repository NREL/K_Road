from k_road.constants import Constants
from k_road.util import *
from .dynamic_single_body_vehicle import DynamicSingleBodyVehicle


class VehicleDDBM(DynamicSingleBodyVehicle):
    """
    Like VehicleDBM, but with differential actions: jerk, steer_angle_rate
    """

    def __init__(self,
                 *args,
                 max_jerk: float = Constants.max_jerk,
                 min_jerk: float = Constants.min_jerk,
                 max_steer_rate: float = Constants.max_steer_rate,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.max_jerk: float = max_jerk
        self.min_jerk: float = min_jerk
        self.max_steer_rate: float = max_steer_rate

    def apply_normalized_control(self,
                                 time_step_length: float,
                                 normalized_jerk: float,
                                 normalized_steer_rate: float) -> None:
        self.apply_control(
            hinge_transform(normalized_jerk, self.min_jerk, self.max_jerk),
            normalized_steer_rate * self.max_steer_rate
        )

    def apply_control(self, time_step_length, jerk, steer_angle_rate):
        jerk = clamp(jerk, self.min_jerk, self.max_jerk)
        acceleration = self.acceleration + jerk * time_step_length

        steer_angle_rate = clamp(steer_angle_rate, -self.max_steer_rate, self.max_steer_rate)
        steer_angle = self.steer_angle + steer_angle_rate * time_step_length

        # if jerk < 0:
        #     print('jerk: ', jerk, ' acc', self.acceleration, ' vel', self.speed)
        # print('jerk ', jerk, ' acc ', self.acceleration, ' sar ', steer_angle_rate, ' sa ', self.steer_angle)
        super().apply_control(time_step_length, acceleration, steer_angle)
