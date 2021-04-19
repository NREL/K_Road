import numpy as np

from factored_gym import ActionTransform
from factored_gym.controller.controller import *
from k_road.scan.waypoint_path_follower import WaypointPathFollower
from k_road.util import *
from scenario.trajectory_tracking.controller.pid_controllers import PIDLongitudinalController


class PurePursuitController(ActionTransform, Controller):
    """
        Preliminary Implementation of Pure Pursuit
        This controller does not control for
            acceleration; only steering
    """

    def __init__(self, look_ahead_distance=1., L=1.04 + 1.42, k=1.,
                 proportional_acc_gain: float = 0.5):
        self.ld = None  # look ahead distance
        self.R = None
        self.alpha = None
        self.kappa = None
        self.L = L  # lf+ld
        self.k = k  # tunable parameter
        self.look_ahead_distance = look_ahead_distance
        self.original_ld = look_ahead_distance
        self.u_optimal = [0, 0]
        self.proportional_acc_gain = proportional_acc_gain

    def get_action(self, process: Process):
        return self.transform_action(process, np.array([0.0, 0.0]))

    def quadrant_correct_angle(self, angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle

    def transform_action(self, process: Process, action):
        ego = process.ego_vehicle
        sim_step = process.time_step_length
        self.L = ego.length
        state = [ego.position[0], ego.position[1], \
                 ego.internal_longitudinal_velocity, \
                 ego.internal_lateral_velocity, ego.angle, \
                 ego.angular_velocity]

        v_x = ego.velocity[0]

        v = np.sqrt(ego.internal_longitudinal_velocity ** 2 + \
                    ego.internal_lateral_velocity ** 2)

        waypoint_path_follower = WaypointPathFollower(
            process.space,
            process.cross_track.point,
            0.1,
            True
        )

        crosstrack_point = process.cross_track.point
        delta_p = process.cross_track.point - ego.position
        future_waypoint = waypoint_path_follower.get_next_waypoint(self.look_ahead_distance)[1]
        self.ld = np.linalg.norm(future_waypoint - ego.position)
        heading_future_waypoint = np.arctan2(future_waypoint[1] - crosstrack_point[1],
                                             future_waypoint[0] - crosstrack_point[0])

        waypoint_angle = signed_delta_angle(ego.velocity.angle, heading_future_waypoint)

        # xL, yL = future_waypoint[0] - ego.position[0], future_waypoint[1] - ego.position[1]

        # ed = process.cross_track.distance
        self.accel_conroller = PIDLongitudinalController(process,
                                                         K_P=self.proportional_acc_gain,
                                                         K_D=0.0,
                                                         K_I=0.0)
        pid_accel = self.accel_conroller.run_step(process.cross_track.shape.body.entity.target_speed)
        delta = np.arctan2(2. * self.L * np.sin(waypoint_angle), self.k * v)
        # print("target: ", process.cross_track.shape.body.entity.target_speed)
        u_optimal = np.array([pid_accel, -delta])
        # print('Pure Pursuit u_optimal', u_optimal)
        self.u_optimal = u_optimal

        return u_optimal
