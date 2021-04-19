from typing import (
    TYPE_CHECKING,
    Tuple,
)

from pymunk import (
    Vec2d,
)
from scipy.integrate import solve_ivp

from k_road.entity.vehicle.dynamic_single_body_vehicle import DynamicSingleBodyVehicle
from k_road.util import *

if TYPE_CHECKING:
    from k_road.k_road_process import KRoadProcess


class VehicleMatlab3DOF(DynamicSingleBodyVehicle):
    """
    Based on the Mathworks single vehicle track 3DOF rigid vehicle body model,
    as described here: https://www.mathworks.com/help/vdynblks/ref/vehiclebody3dof.html
    """

    def __init__(
            self,
            parent: 'KRoadProcess',
            color: Tuple[int, int, int],
            position: Vec2d,
            body_velocity: Vec2d,
            yaw: float,
            # mass: float = 2000.,  # [kg] Falcone 2007
            # yaw_inertia: float = 4000.,  # [kg/m^2]
            # front_axel_distance: float = 1.4,  # distance of front axel from center of mass
            # rear_axel_distance: float = 1.6,  # distance of rear axel from center of mass
            road_friction_coefficient: float = 1.0,  # .9 (dry road), .6 (wet road), .2 (snow), .05 (ice)
            # center_of_mass_height: float = 0.35,  # [m]
            # front_tire_cornering_stiffness: float = 19e3,  # [N/rad]
            # rear_tire_cornering_stiffness: float = 33.5e3,  # [N/rad]
            # nominal_normal_force: float = 5000,  # [N] Nominal downward force
            tire_model: any = None,  # not used
            **kwargs
    ):
        super().__init__(parent, position, body_velocity, yaw, color, **kwargs)

        self.collided: bool = False

        self.g: float = 9.81

        # self.mass: float = mass
        # self.moment: float = yaw_inertia
        # self.front_axel_distance: float = front_axel_distance
        # self.rear_axel_distance: float = rear_axel_distance
        # self.road_friction_coefficient: float = road_friction_coefficient
        # self.center_of_mass_height = center_of_mass_height
        # self.nominal_normal_force = nominal_normal_force
        # # self.nominal_normal_force: float = (self.mass / 4.0) * self.g  # approximate downward force on each tire
        # self.front_tire_cornering_stiffness = front_tire_cornering_stiffness
        # self.rear_tire_cornering_stiffness = rear_tire_cornering_stiffness

        self.mass: float = 2000.0
        self.moment: float = 4000.0
        self.front_axel_distance: float = 1.4
        self.rear_axel_distance: float = 1.6
        self.road_friction_coefficient: float = 1.0
        self.center_of_mass_height = .35
        self.F_znom = 5000.0
        # self.nominal_normal_force = self.mass * self.g / 4.0
        self.front_tire_cornering_stiffness = 19e3
        self.rear_tire_cornering_stiffness = 33.5e3
        self.tire_saturation: float = 70000.0

        self.internal_angular_velocity: float = self.angular_velocity
        self.internal_longitudinal_velocity: float = self.body_velocity.x
        self.internal_lateral_velocity: float = self.body_velocity.y

        self.previous_position = self.position

    @property
    def instantaneous_angular_velocity(self) -> float:
        return self.internal_angular_velocity

    @property
    def instantaneous_body_velocity(self) -> Vec2d:
        return Vec2d(self.internal_longitudinal_velocity, self.internal_lateral_velocity)

    @property
    def instantaneous_global_velocity(self) -> Vec2d:
        return self.instantaneous_body_velocity.rotated(self.angle)

    def is_in_valid_state(self) -> bool:
        return \
            fabs(signed_delta_angle(self.angle, self.velocity.angle)) < (60.0 * pi / 180.0) and \
            fabs(self.internal_angular_velocity) < (180.0 * pi / 180.0)

    def apply_control(self, time_step_length: float, acceleration: float, steer_angle: float) -> None:
        self.previous_position = self.position
        acceleration = clamp(acceleration, self.max_deceleration, self.max_acceleration)
        steer_angle = clamp(steer_angle, -self.max_steer_angle, self.max_steer_angle)

        # print('apply_control {} {} {}'.format(acceleration, steer_angle, time_step_length))

        self.steer_angle = steer_angle
        self.acceleration = acceleration

        # self.internal_longitudinal_velocity += acceleration * time_step_length
        # self.internal_longitudinal_velocity = clamp(self.internal_longitudinal_velocity, 0.0, self.max_speed)

        ode_state = [
            0,  # reference x
            0,  # reference y
            0.0,  # reference angle
            self.internal_longitudinal_velocity,
            self.internal_lateral_velocity,
            self.internal_angular_velocity]
        aux_state = (steer_angle, acceleration)
        delta_ode_state = solve_ivp(self._integrator, [0, time_step_length], ode_state, args=aux_state)

        dx, dy, dangle, \
        self.internal_longitudinal_velocity, \
        self.internal_lateral_velocity, \
        self.internal_angular_velocity = delta_ode_state.y[:, -1]

        self.set_velocity_and_yaw(
            Vec2d(self.internal_longitudinal_velocity, self.internal_lateral_velocity), self.angle)

        # set pymunk body parameters to match target position after one physics timestep
        self.angular_velocity = dangle / time_step_length
        self.velocity = Vec2d(dx, dy).rotated(self.angle) / time_step_length

    def _integrator(self,
                    t: float,
                    state,
                    steer_angle: float,
                    acceleration: float,
                    ):
        x, y, yaw, longitudinal_velocity, lateral_velocity, angular_velocity = state

        # front_longitudinal_force = 0.0
        front_lateral_force = 0.0
        # rear_longitudinal_force = 0.0
        rear_lateral_force = 0.0

        if fabs(longitudinal_velocity) >= 1e-2:
            longitudinal_moment = \
                (acceleration - lateral_velocity * angular_velocity) * self.mass * self.center_of_mass_height
            effective_length = self.front_axel_distance + self.rear_axel_distance
            F_z = self.mass * self.g

            F_zf = max(0.0, (self.rear_axel_distance * F_z - longitudinal_moment) / effective_length)

            F_zr = max(0.0, (self.front_axel_distance * F_z + longitudinal_moment) / effective_length)

            adjusted_steering_angle = steer_angle * tanh(4 * longitudinal_velocity)
            alpha_f = signed_delta_angle(atan2(lateral_velocity + self.front_axel_distance * angular_velocity,
                                               longitudinal_velocity), adjusted_steering_angle)

            F_zf_relative = F_zf / self.F_znom
            front_tire_lateral_force = - self.front_tire_cornering_stiffness * alpha_f * \
                                       self.road_friction_coefficient * F_zf_relative

            F_zf_tire_saturation_scale = self.tire_saturation * F_zf_relative
            front_tire_lateral_saturation = \
                clamp(front_tire_lateral_force, -F_zf_tire_saturation_scale, F_zf_tire_saturation_scale)

            F_zr_relative = F_zr / self.F_znom
            alpha_r = atan2(lateral_velocity - self.rear_axel_distance * angular_velocity,
                            longitudinal_velocity)
            rear_tire_lateral_force = -self.rear_tire_cornering_stiffness * alpha_r * \
                                      self.road_friction_coefficient * F_zr_relative
            # F_zr_tire_saturation_scale = self.tire_saturation * F_zr_relative
            # rear_tire_lateral_saturation = self.saturate_tire(rear_tire_lateral_force,F_zr_tire_saturation_scale)

            # front_longitudinal_force = -front_tire_lateral_force * sin(steer_angle)
            # front_lateral_force = front_tire_lateral_force * cos(steer_angle) \
            #                       + front_tire_lateral_saturation * sin(steer_angle) * sin(steer_angle)
            front_lateral_force = front_tire_lateral_force * (cos(steer_angle) + sin(steer_angle) ** 2)
            # rear_longitudinal_force = 0.0
            rear_lateral_force = rear_tire_lateral_force

        # longitudinal_acceleration = 0

        lateral_acceleration = (-longitudinal_velocity * angular_velocity) + \
                               (front_lateral_force + rear_lateral_force) / self.mass

        angular_acceleration = (self.front_axel_distance * front_lateral_force -
                                self.rear_axel_distance * rear_lateral_force) / self.moment

        body_velocity = Vec2d(longitudinal_velocity, lateral_velocity)
        velocity = body_velocity.rotated(yaw)

        if longitudinal_velocity >= self.max_speed:
            acceleration = longitudinal_velocity - self.max_speed

        # print('{} {} {}'.format(longitudinal_velocity, lateral_velocity, angular_acceleration))
        return [velocity.x,
                velocity.y,
                angular_velocity,
                acceleration,
                lateral_acceleration,
                angular_acceleration]
