from typing import (
    Optional,
    TYPE_CHECKING,
    Tuple,
)

import numpy as np
from pymunk import (
    Vec2d,
)
from scipy.integrate import odeint

from k_road.constants import Constants
from k_road.entity.vehicle.dynamic_single_body_vehicle import DynamicSingleBodyVehicle
from k_road.util import *

if TYPE_CHECKING:
    from k_road.k_road_process import KRoadProcess


class VehicleKBM(DynamicSingleBodyVehicle):
    """
    Using the kinematic bicycle model.
    """

    def __init__(
            self,
            parent: 'KRoadProcess',
            color: Tuple[int, int, int],
            position: Vec2d,
            body_velocity: Vec2d,
            yaw: float,
            front_wheel_spacing: Optional[float] = None,
            rear_wheel_spacing: Optional[float] = None,
            **kwargs
    ):
        super().__init__(parent, position, body_velocity, yaw, color, **kwargs)

        self.collided: bool = False

        # distance of front wheels from center of mass
        self.front_wheel_spacing: float = .8 * self.length / 2 if front_wheel_spacing is None else front_wheel_spacing

        # distance of rear wheels from center of mass
        self.rear_wheel_spacing: float = .8 * self.length / 2 if rear_wheel_spacing is None else rear_wheel_spacing

        self.acceleration: float = 0
        self.previous_position = self.position

    def apply_normalized_control_action(self,
                                        time_step_length: float,
                                        action) -> None:
        """
        applies the normalized action from an action tuple / matrix
        """
        return self.apply_normalized_control(time_step_length, action[0], action[1])

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

    def predict_position(self, time_step_length: float, acceleration, steer_angle) -> list:
        """
            Same as apply_control but without actually applying the control

            Returns future states
        """
        previous_position = self.position

        acceleration = Util.clamp(acceleration, self.max_deceleration, self.max_acceleration)
        steer_angle = Util.clamp(steer_angle, -self.max_steer_angle, self.max_steer_angle)

        t = time_step_length

        ode_state = [
            0, 0,
            0,
            self.velocity.length]
        aux_state = (steer_angle, acceleration)
        step = t / 2.0
        steps = np.arange(0.0, t + step, step)
        delta_ode_state = odeint(self.__integrator, ode_state, steps, args=aux_state)
        longitudinal_delta, lateral_delta, angle_delta, speed = delta_ode_state[-1]
        longitudinal_delta = max(0.0, longitudinal_delta)

        longitudinal_velocity = longitudinal_delta / time_step_length
        lateral_velocity = lateral_delta / time_step_length
        angular_velocity = angle_delta / time_step_length

        return [previous_position[0] + longitudinal_delta, previous_position[1] + lateral_delta, \
                longitudinal_velocity, lateral_velocity, self.yaw + angle_delta, angular_velocity]

    def apply_control(self, time_step_length: float, acceleration: float, steer_angle: float) -> None:
        """
            Kinematic bicycle model: https://borrelli.me.berkeley.edu/pdfpub/IV_KinematicMPC_jason.pdf

            x' = v * cos(phi + beta)
            y' = v * sin(phi + beta)
            phi' = (v/lr)*sin(beta)
            v' = a
            beta = atan((lr/(lf+lr))*tan(delta))
        """

        self.previous_position = self.position

        # print('acc: ', acceleration, ' steer: ', steer_angle)

        acceleration = clamp(acceleration, self.max_deceleration, self.max_acceleration)
        steer_angle = clamp(steer_angle, -self.max_steer_angle, self.max_steer_angle)

        self.steer_angle = steer_angle
        self.acceleration = acceleration

        t = time_step_length

        ode_state = [
            0, 0,
            0,
            self.velocity.length]
        aux_state = (steer_angle, acceleration)
        # steps = np.arange(0.0, t, t / 10.0)
        step = t / 2.0
        steps = np.arange(0.0, t + step, step)
        delta_ode_state = odeint(self.__integrator, ode_state, steps, args=aux_state)
        longitudinal_delta, lateral_delta, angle_delta, speed = delta_ode_state[-1]
        longitudinal_delta = max(0.0, longitudinal_delta)

        longitudinal_velocity = longitudinal_delta / time_step_length
        lateral_velocity = lateral_delta / time_step_length
        angular_velocity = angle_delta / time_step_length

        self.angular_velocity = angular_velocity
        self.set_velocity_and_yaw(Vec2d(longitudinal_velocity, lateral_velocity), self.angle)

    def __integrator(self, state, t: float, steer_angle: float, acceleration: float):
        x, y, yaw, v = state

        # velocity heading relative to body
        beta = math.atan(
            (self.rear_wheel_spacing / (self.front_wheel_spacing + self.rear_wheel_spacing)) * math.tan(steer_angle))

        d_x = v * math.cos(steer_angle + beta)
        d_y = v * math.sin(steer_angle + beta)
        d_yaw = (v / self.rear_wheel_spacing) * math.sin(beta)
        d_v = acceleration

        return [d_x, d_y, d_yaw, d_v]
