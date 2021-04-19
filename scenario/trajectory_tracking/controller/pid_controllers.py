"""
    Original code written by Monte Lunacek

"""
import math
from collections import deque

import numpy as np

from factored_gym import ActionTransform
from factored_gym.controller.controller import *
from k_road.k_road_process import KRoadProcess
from k_road.scan.waypoint_path_follower import WaypointPathFollower


class PIDLongitudinalController(object):

    def __init__(self, process: KRoadProcess, K_P=1.0, K_D=0.0, K_I=0.0):

        self._vehicle = process.ego_vehicle
        self.process = process
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        self._dt = process.time_step_length
        self._e_buffer = deque(maxlen=30)

    def run_step(self, target_speed, debug=False):
        current_speed = self._vehicle.instantaneous_body_velocity.length
        # current_speed = self._vehicle.internal_longitudinal_velocity
        target_speed = self.process.cross_track.shape.body.entity.target_speed

        if debug:
            print('Current speed = {}'.format(current_speed))

        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle of the vehicle based on the PID equations
        :param target_speed:  target speed in Km/h
        :param current_speed: current speed of the vehicle in Km/h
        :return: throttle control in the range [0, 1]
        """
        _e = (target_speed - current_speed)
        self._e_buffer.append(_e)

        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._K_P * _e) + (self._K_D * _de / self._dt) + \
                       (self._K_I * _ie * self._dt), -10.0, 10.0)


class PIDLateralController(object):

    def __init__(self, process: Process, K_P=1.0, K_D=0.0, K_I=0.0):

        self._vehicle = process.ego_vehicle
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        self._dt = process.time_step_length
        self._e_buffer = deque(maxlen=10)

    def run_step(self, waypoint, debug=False):
        return self._pid_control(waypoint)

    def _pid_control(self, waypoint):
        """
        Estimate the steering angle of the vehicle based on the PID equations
        :param waypoint: target waypoint
        :param vehicle_transform: current transform of the vehicle
        :return: steering control in the range [-1, 1]
        """

        v_begin = self._vehicle.position
        yaw_x = math.cos(self._vehicle.angle)
        yaw_y = math.sin(self._vehicle.angle)
        v_end = [v_begin[0] + yaw_x, v_begin[1] + yaw_y]

        v_vec = np.array([v_end[0] - v_begin[0], v_end[1] - v_begin[1], 0.0])
        w_vec = np.array([waypoint[0] - v_begin[0], waypoint[1] - v_begin[1], 0.0])
        _dot = math.acos(np.clip(np.dot(w_vec, v_vec) /
                                 (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

        _cross = np.cross(v_vec, w_vec)

        if _cross[2] < 0:
            _dot *= -1.0

        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        value = np.clip((self._K_P * _dot) + (self._K_D * _de /
                                              self._dt) + (self._K_I * _ie * self._dt), -1.0, 1.0)

        return value


class VehiclePIDController(ActionTransform, Controller):

    def __init__(self, args_lateral=None, args_longitudinal=None, look_ahead_multiplier=1.0, verbose=False):

        if args_lateral is None:
            self.args_lateral = {'K_P': 0.1, 'K_D': 0.0, 'K_I': 0.0}
        else:
            self.args_lateral = args_lateral

        if args_longitudinal is None:
            self.args_longitudinal = {'K_P': .5, 'K_D': 0.0, 'K_I': 0.0}
        else:
            self.args_longitudinal = args_longitudinal

        self.verbose = verbose
        self.ld = look_ahead_multiplier
        self.future_crosstrack = None

    def get_action(self, process: Process):
        return self.transform_action(process, np.array([0.0, 0.0]))

    def transform_action(self, process: Process, action):

        self._lon_controller = PIDLongitudinalController(process, \
                                                         **self.args_longitudinal)
        self._lat_controller = PIDLateralController(process, \
                                                    **self.args_lateral)

        waypoint_path_follower = WaypointPathFollower(
            process.space,
            process.cross_track.point,
            10,
            True,
            path=process.cross_track.shape.body.entity
        )

        # future_crosstrack = waypoint_path_follower.get_next_waypoint(self.ld)[1]
        ld = self.ld * process.ego_vehicle.internal_longitudinal_velocity * process.time_step_length
        future_crosstrack = waypoint_path_follower.get_next_waypoint(ld)[1]
        self.future_crosstrack = future_crosstrack

        # waypoint = np.array([process.cross_track.point.x, \
        #                                process.cross_track.point.y])

        waypoint = np.array([future_crosstrack[0], future_crosstrack[1]])
        [throttle, steering] = self.run_step(waypoint=waypoint, action=action)
        if self.verbose:
            print("PID u_optimal: ", [throttle, steering])
        return np.array([throttle, steering])

    def run_step(self, waypoint, target_speed=25, debug=False, action=[0, 0]):
        throttle, steering = action
        throttle += self._lon_controller.run_step(target_speed, debug)
        steering += self._lat_controller.run_step(waypoint, debug)

        return [throttle, steering]

    def render(self, process: 'Process', view: 'KRoadView') -> None:
        if self.future_crosstrack is not None:
            view.draw_circle((200, 0, 255), self.future_crosstrack, .6)
