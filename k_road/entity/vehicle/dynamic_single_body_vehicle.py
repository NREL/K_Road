from abc import abstractmethod

from k_road.constants import Constants
from k_road.entity.vehicle.single_body_vehicle import SingleBodyVehicle
from k_road.util import (
    clamp,
    hinge_transform,
)


class DynamicSingleBodyVehicle(SingleBodyVehicle):
    """
    A single-body vehicle that can take an action
    """

    def __init__(self,
                 *args,
                 **kwargs
                 ) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def apply_control(self, time_step_length: float, acceleration: float, steer_angle: float) -> None:
        pass

    def apply_normalized_control_action(self,
                                        time_step_length: float,
                                        action) -> None:
        """
        applies the normalized action from an action tuple / matrix
        """
        # print('control: ', action)
        return self.apply_normalized_control(time_step_length, clamp(action[0]), clamp(action[1]))

    def apply_normalized_control(self,
                                 time_step_length: float,
                                 normalized_acceleration: float,
                                 normalized_steer_angle: float) -> None:
        """
        :param normalized_acceleration acceleration - between -1 (full brake), 0 (no input), and 1 (full throttle)
        :param normalized_steer_angle - between -1 (full left), 0 (forward), and 1 (full right)
        """
        acceleration = hinge_transform(normalized_acceleration, self.max_deceleration, self.max_acceleration)
        steer_angle = normalized_steer_angle * Constants.max_steer_angle
        return self.apply_control(time_step_length, acceleration, steer_angle)
