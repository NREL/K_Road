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

from k_road.entity.vehicle.dynamic_single_body_vehicle import DynamicSingleBodyVehicle
from k_road.model.tire_model.lateral_tire_model import LateralTireModel
from k_road.model.tire_model.linear_tire_model import LinearTireModel
from k_road.util import *

if TYPE_CHECKING:
    from k_road.k_road_process import KRoadProcess


class VehicleDBM3(DynamicSingleBodyVehicle):
    """
    Uses a linear tire model coupled with a dynamic bicycle model. Not based on FORDS implementation.

    vehicle model implementation: https://github.com/MPC-Berkeley/barc/blob/master/workspace/src/barc/src/estimation
    /system_models.py
    FORDS/FODS: http://www.me.berkeley.edu/~frborrel/pdfpub/IV_KinematicMPC_jason.pdf
    FORDS source: https://github.com/naterarmstrong/gym-driver
        dynamic source: https://github.com/naterarmstrong/gym-driver/blob/master/gym_driver/src/DynamicCar.py
    cornering stiffness: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.46.28&rep=rep1&type_=pdf
    yaw inertia: https://www.degruyter.com/downloadpdf/j/mecdc.2013.11.issue-1/mecdc-2013-0003/mecdc-2013-0003.pdf
    friction coefficients: http://www.gipsa-lab.grenoble-inp.fr/~moustapha.doumiati/MED2010.pdf

        + sprung and unspung masses (suspension model)
        + torque and braking response
    """

    def __init__(
            self,
            parent: 'KRoadProcess',
            color: Tuple[int, int, int],
            position: Vec2d,
            body_velocity: Vec2d,
            yaw: float,
            # mass: float = 1500,  # Toyota Camry gross mass (kg)
            mass: float = 2000.,  # [kg] Falcone 2007
            # yaw_inertia: float = 2250,
            # yaw_inertia: float = 3344.,  # [kg/m^2] Falcone 2007
            yaw_inertia: float = 4000.,  # [kg/m^2]
            front_wheel_spacing: float = 1.4,
            rear_wheel_spacing: float = 1.6,
            road_friction_coefficient: float = .9,  # .9 (dry road), .6 (wet road), .2 (snow), .05 (ice)
            tire_model: Optional[LateralTireModel] = None,
            **kwargs
    ):
        super().__init__(parent, position, body_velocity, yaw, color, **kwargs)

        self.mass: float = mass
        self.moment: float = yaw_inertia

        self.collided: bool = False

        # distance of front wheels from center of mass
        self.front_wheel_spacing: float = .8 * self.length / 2 if front_wheel_spacing is None else front_wheel_spacing

        # distance of rear wheels from center of mass
        self.rear_wheel_spacing: float = .8 * self.length / 2 if rear_wheel_spacing is None else rear_wheel_spacing

        self.road_friction_coefficient: float = road_friction_coefficient

        if tire_model is None:
            tire_model = LinearTireModel(mu=road_friction_coefficient)
            tire_model.estimate_stiffness_from_mass_and_spacing(
                self.mass,
                self.front_wheel_spacing,
                self.rear_wheel_spacing)
            # self.tire_model2 = LinearTireModel(mu=road_friction_coefficient)
            # Ca = self.tire_model2.estimate_stiffness_from_mass_and_spacing(
            #     self.mass,
            #     self.front_wheel_spacing,
            #     self.rear_wheel_spacing)
            #
            # tire_model = FialaBrushTireModel()
            # print('ca: ', tire_model.Ca - Ca, tire_model.Ca, Ca)

        self.tire_model: LateralTireModel = tire_model

        self.internal_angular_velocity: float = self.angular_velocity
        self.internal_longitudinal_velocity: float = self.body_velocity.x
        self.internal_lateral_velocity: float = self.body_velocity.y

        self.front_tire_lateral_force = 0.0
        self.rear_tire_lateral_force = 0.0

        self.alt_front_tire_lateral_force = 0.0
        self.alt_rear_tire_lateral_force = 0.0

        self.previous_position = self.position

        self.target_position = self.position
        self.target_angle = self.angle

        self.rear_tire_lateral_force = None
        self.front_tire_lateral_force = None

        self.Fz: float = (self.mass / 4.0) * 9.81  # approximate downward force on each tire

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
        # fabs(self.internal_lateral_velocity) < 20.0

    def apply_control(self, time_step_length: float, acceleration: float, steer_angle: float) -> None:
        """
            Dynamic bicycle model: https://borrelli.me.berkeley.edu/pdfpub/IV_KinematicMPC_jason.pdf

            x'' = phi' * y' + acc
            y'' = -phi * x' + (2/m) * (Fcf * cos(steer_angle) + Fcr)
            phi'' = (2/Iz) * (lf * Fcf - lr * Fcr)
            Fcf = - road_friction_coefficient * m * (lr / (lf + lr))
            Fcr = - road_friction_coefficient * m * (lf / (lf + lr))

        """
        # print('vehicle apply_control', time_step_length, acceleration, steer_angle)
        # if random.randrange(5000) == 0:
        #    print('st: ',
        #          self.target_angle - self.angle,
        #          self.target_position - self.position,
        #          self.internal_angular_velocity - self.angular_velocity,
        #          self.internal_longitudinal_velocity,
        #          self.internal_lateral_velocity,
        #          self.internal_angular_velocity,
        #          self.angle,
        #          self.velocity,
        #          self.body_velocity)

        self.previous_position = self.position

        print('acc: ', acceleration, ' steer: ', steer_angle)

        acceleration = clamp(acceleration, self.max_deceleration, self.max_acceleration)
        steer_angle = clamp(steer_angle, -self.max_steer_angle, self.max_steer_angle)

        self.steer_angle = steer_angle
        self.acceleration = acceleration

        t = time_step_length

        ode_state = [
            0,  # reference x
            0,  # reference y
            self.internal_longitudinal_velocity,
            self.internal_lateral_velocity,
            self.angle,
            self.internal_angular_velocity]

        aux_state = (steer_angle, acceleration)

        # steps = np.linspace(0.0, time_step_length, 3, endpoint=True)
        steps = np.linspace(0.0, time_step_length, 12, endpoint=True)
        # print('steps {} {}'.format(steps, self.internal_longitudinal_velocity))

        initial = self.internal_longitudinal_velocity

        delta_ode_state = odeint(self._integrator, ode_state, steps, args=aux_state)

        dx, dy, self.internal_longitudinal_velocity, self.internal_lateral_velocity, \
        angle, self.internal_angular_velocity = delta_ode_state[-1]

        # print('delta {} {}'.format(self.internal_longitudinal_velocity,
        #                            (self.internal_longitudinal_velocity - initial) / time_step_length))

        # clamp velocity
        self.internal_longitudinal_velocity = clamp(self.internal_longitudinal_velocity, 0.0, self.max_speed)

        self.set_velocity_and_yaw(
            Vec2d(self.internal_longitudinal_velocity, self.internal_lateral_velocity), self.angle)

        # set pymunk body parameters to match target position after one physics timestep
        self.angular_velocity = signed_delta_angle(angle, self.angle) / time_step_length

        self.velocity = Vec2d(dx, dy) / time_step_length

        self.target_position = self.position + Vec2d(dx, dy)
        self.target_angle = angle

    def _integrator(self,
                    state,
                    t: float,
                    steer_angle: float,
                    acceleration: float):
        '''
        https://www.researchgate.net/publication/271545759_The_3
        -DoF_bicycle_model_with_the_simplified_piecewise_linear_tire_model
        mass * (longitudinal_acceleration - lateral_velocity * angular_velocity) = Fxf + Fxr
        longitudinal_acceleration = (Fxf + Fxr) / (mass) + lateral_velocity * angular_velocity
        longitudinal_acceleration = (forward_acceleration) + lateral_velocity * angular_velocity


        (lateral_acceleration + longitudinal_velocity * angular_velocity) = Fyf + Fyr
        lateral_acceleration  = Fyf + Fyr - longitudinal_velocity * angular_velocity
        lateral_acceleration  = (2 / mass) * (Fyf * cos(steer_angle) + Fyr) - longitudinal_velocity * angular_velocity
            ^^ (x2 from two sets of wheels)

        moment * angular_acceleration  = Fyf * front_wheel_spacing - Fyr * rear_wheel_spacing
        angular_acceleration = (Fyf * front_wheel_spacing - Fyr * rear_wheel_spacing) / moment
        angular_acceleration = (2 / moment) * (Fyf * front_wheel_spacing - Fyr * rear_wheel_spacing)
            ^^ (x2 from two sets of wheels)

        Fxf = Fxwf * cos(steer_f) - Fywf * sin(steer_f)
        Fyf = Fxwf * sin(steer_f) + Fywf * cos(steer_f)
        Fxr = Fxwr * cos(steer_r) - Fywr * sin(steer_r) = Fxwr
        Fyr = Fxwr * sin(steer_r) + Fywr * cos(steer_r) = Fywr
        '''

        x, y, longitudinal_velocity, lateral_velocity, yaw, angular_velocity = state

        # if fabs(longitudinal_velocity) <= 1e-3 and fabs(acceleration) <= 0:
        #     acceleration = -longitudinal_velocity

        '''
        Slip angles
         From http://code.eng.buffalo.edu/dat/sites/model/bicycle.html
         FORDS uses a different formula:
         beta = np.arctan((self.l_r / (self.l_f + self.l_r)) * np.tan(delta_f))
         vel = np.sqrt(dx_body ** 2 + dy_body ** 2)
         slip_angle = (vel / self.l_r) * np.sin(beta)

        see Hindiyeh's Thesis, p38 (https://ddl.stanford.edu/sites/g/files/sbiybj9456/f/publications
        /2013_Thesis_Hindiyeh_Dynamics_and_Control_of_Drifting_in_Automobiles.pdf)

        another approximation is:
        Tire Model:

         u_wf = longitudinal_velocity * cos(steer_angle) + (lateral_velocity + front_wheel_spacing * angular_velocity)
            *sin(steer_angle)
         u_wr = longitudinal_velocity * cos(steer_angle) + (lateral_velocity - rear_wheel_spacing * angular_velocity)
            *sin(steer_angle)

         front_slip_angle = steer_angle - lateral_velocity + front_wheel_spacing * angular_velocity / u_wf
         rear_slip_angle = (rear_wheel_spacing * angular_velocity - lateral_velocity) / u_wr
        '''
        # front_slip_angle = signed_delta_angle(
        #     atan2(lateral_velocity + self.front_wheel_spacing * angular_velocity, longitudinal_velocity),
        #     steer_angle)
        # self.front_slip_angle = front_slip_angle
        #
        # rear_slip_angle = \
        #     atan2(lateral_velocity - self.rear_wheel_spacing * angular_velocity, longitudinal_velocity)
        # self.rear_slip_angle = rear_slip_angle
        sin_steer_angle = sin(steer_angle)
        cos_steer_angle = cos(steer_angle)

        # tire cornering stiffness estimate
        # from FORDS: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2017/EECS-2017-102.pdf
        # tire_cornering_stiffness = road_friction_coefficient * self.mass * (
        #         self.rear_wheel_spacing / (self.front_wheel_spacing + self.rear_wheel_spacing))

        # Linear tire model, lateral forces aka cornering forces:
        # http://code.eng.buffalo.edu/dat/sites/model/linear.html
        # https://borrelli.me.berkeley.edu/pdfpub/IV_KinematicMPC_jason.pdf
        # Fyf = -front_tire_cornering_stiffness * front_slip_angle
        # Fyr = -rear_tire_cornering_stiffness * rear_slip_angle
        # front_tire_lateral_force = -2 * tire_cornering_stiffness * front_slip_angle
        # rear_tire_lateral_force = -2 * tire_cornering_stiffness * rear_slip_angle

        # approximate longitudinal force on front wheels
        Fx0 = (acceleration * (self.mass / 2))

        # # approximate Vc_front with zero longitudinal slip
        # Vc_front = longitudinal_velocity * cos(front_slip_angle) + lateral_velocity * sin(front_slip_angle)
        #
        # self.alt_front_tire_lateral_force = 2 * self.tire_model.get_lateral_force(
        #     self.Fz,
        #     front_slip_angle,
        #     Fx0,
        #     Vc_front,
        #     0.0,
        #     0.0
        # )[0]
        #
        # # approximate Vc_rear with zero longitudinal slip
        # Vc_rear = longitudinal_velocity * cos(rear_slip_angle) + lateral_velocity * sin(rear_slip_angle)
        #
        # self.alt_rear_tire_lateral_force = 2 * self.tire_model.get_lateral_force(
        #     self.Fz,
        #     rear_slip_angle,
        #     0.0,
        #     Vc_rear,
        #     0.0,
        #     0.0
        # )[0]

        # Cf = 1200.0 * 2.0  # N/rad
        # Cr = 1100.0 * 2.0  # N/rad

        # Cf = 12e3 * 2.0  # N/rad
        # Cr = 11e3 * 2.0  # N/rad
        Cf = 19.0e3 * 2.0  # N/rad
        Cr = 33.5e3 * 2.0  # N/rad
        if fabs(longitudinal_velocity) < 1e-1:
            self.front_tire_lateral_force = 0
            self.rear_tire_lateral_force = 0
        else:
            self.front_tire_lateral_force = -Cf * atan2(
                ((lateral_velocity + self.front_wheel_spacing * angular_velocity) /
                 longitudinal_velocity - steer_angle),
                1.0)
            self.rear_tire_lateral_force = -Cr * atan2(
                (lateral_velocity - self.rear_wheel_spacing * angular_velocity) / longitudinal_velocity, 1.0)

        # print("{} {} {} | {} {} {} |".format(self.alt_front_tire_lateral_force, self.front_tire_lateral_force,
        #                                      self.alt_front_tire_lateral_force - self.front_tire_lateral_force,
        #                                      self.alt_rear_tire_lateral_force, self.rear_tire_lateral_force,
        #                                      self.alt_rear_tire_lateral_force - self.rear_tire_lateral_force))

        # acceleration
        # longitudinal_acceleration = acceleration + lateral_velocity * angular_velocity \
        #                             + (1 / self.mass) * (-front_tire_lateral_force * sin(steer_angle))
        longitudinal_acceleration = acceleration

        # c_a = 1.36  # aerodynamic coefficient
        # c_a = .5 * 1.29 * .3 * 2  # aerodynamic coefficient
        # c_a = 0.0
        # c_r1 = 0.10  # 1 - friction coefficient
        #
        # R_x = c_r1 * fabs(self.longitudinal_velocity)
        # F_aero = c_a * self.longitudinal_velocity ** 2
        # F_load = F_aero + R_x
        # longitudinal_acceleration = acceleration \
        #                             - (F_load + self.front_tire_lateral_force * sin_steer_angle) / self.mass \
        #                             + lateral_velocity * angular_velocity
        # longitudinal_acceleration = acceleration

        # - lateral_velocity * angular_velocity
        # - (self.front_tire_lateral_force * sin_steer_angle + F_load) / self.mass

        # longitudinal_acceleration = acceleration
        # throttle - Ffy * math.sin(delta) / m - F_load/m + self.vy * self.omega

        # longitudinal_acceleration = acceleration + lateral_velocity * angular_velocity
        # dynamic_friction = road_friction_coefficient * .1
        # rolling_friction = 0
        # longitudinal_acceleration = acceleration + lateral_velocity * angular_velocity \
        #                             + (1 / self.mass) * (-front_tire_lateral_force * sin(steer_angle)) \
        #                             - dynamic_friction * longitudinal_velocity ** 2 \
        #                             - (rolling_friction if longitudinal_velocity > 0 else 0)

        lateral_acceleration = (self.front_tire_lateral_force * cos_steer_angle + self.rear_tire_lateral_force) \
                               / self.mass \
                               - longitudinal_velocity * angular_velocity

        # longitudinal_acceleration = sqrt(max(0.0, acceleration ** 2 - lateral_acceleration ** 2))
        # if acceleration < 0:
        #     longitudinal_acceleration *= -1

        angular_acceleration = (self.front_tire_lateral_force * self.front_wheel_spacing * cos_steer_angle -
                                self.rear_tire_lateral_force * self.rear_wheel_spacing) / self.moment

        body_velocity = Vec2d(longitudinal_velocity, lateral_velocity)
        velocity = body_velocity.rotated(yaw)

        # print('{} {} {}'.format(longitudinal_velocity, lateral_velocity, angular_acceleration))
        return [velocity.x,
                velocity.y,
                longitudinal_acceleration,
                lateral_acceleration,
                angular_velocity,
                angular_acceleration]
