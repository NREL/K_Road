import numpy as np

from factored_gym import ActionTransform
from factored_gym.controller.controller import *
from k_road.k_road_view import KRoadView
from k_road.util import *
from scenario.trajectory_tracking.controller.pid_controllers import PIDLongitudinalController


class StanleyController(ActionTransform, Controller):
    """
        Preliminary Implementation of Pure Pursuit
        This controller does not control for
            acceleration; only steering

      April 1st, 2020 Edit:
      http://ai.stanford.edu/~gabeh/papers/hoffmann_stanley_control07.pdf
    """

    def __init__(self, k: float = .5, ksoft: float = 1., kdsteer=0., max_steer=np.pi / 3.,
                 k_p_acc: float = 1.0):
        self.k = k
        self.ksoft = ksoft
        self.kdsteer = kdsteer
        self.max_steer = [-max_steer, max_steer]
        self.delta_i = 0.0
        self.delta_im1 = 0.0
        self.k_p_acc = k_p_acc

    def get_action(self, process: Process):
        return self.transform_action(process, np.array([0.0, 0.0]))

    def normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle

    def render(self, process: Process, view: KRoadView):
        pass

    def transform_action(self, process: Process, action):
        ego = process.ego_vehicle

        path_angle = process.cross_track.shape.body.entity.angle

        car_angle = ego.angle

        v = np.sqrt(ego.internal_longitudinal_velocity ** 2 + \
                    ego.internal_lateral_velocity ** 2)
        mass = ego.mass
        lf = ego.front_axel_distance
        lr = ego.rear_axel_distance
        # L = lf + lr

        # Cy = ego.tire_model.estimate_stiffness_from_mass_and_spacing(mass, lf, lr)
        Cy = ego.front_tire_cornering_stiffness
        if self.max_steer is None:
            self.max_steer = [-ego.max_steer_angle, ego.max_steer_angle]

        rtraj = process.cross_track.shape.body.entity.angular_velocity
        # rego = ego.angular_velocity

        kag = Cy / (1. + lf / lr)
        Phiss = kag * ego.speed * rtraj
        Phi = self.normalize_angle(car_angle - path_angle)

        delta_p = process.cross_track.point - ego.position
        cross_track_heading = signed_delta_angle(ego.velocity.angle, delta_p.angle)
        crosstrack_error = np.sign(cross_track_heading) * process.cross_track.distance

        delta = (Phi - Phiss) + np.arctan2(self.k * crosstrack_error,
                                           self.ksoft + ego.internal_longitudinal_velocity) + \
                self.kdsteer * (self.delta_i - self.delta_im1)
        self.delta_im1 = self.delta_i
        self.delta_i = delta
        self.accel_controller = PIDLongitudinalController(process,
                                                          K_P=self.k_p_acc,
                                                          K_D=0.0,
                                                          K_I=0.0)
        # print("target speed: ", process.cross_track.shape.body.entity.target_speed)
        pid_accel = self.accel_controller.run_step(process.cross_track.shape.body.entity.target_speed)
        # print("speed: ", ego.internal_longitudinal_velocity)
        u_optimal = np.array([pid_accel, -delta])
        # print("Stanley + PID u_optimal : ", u_optimal)
        return u_optimal
