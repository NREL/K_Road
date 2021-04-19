"""
    Original code written by Monte Lunacek

"""

import numpy as np

from k_road import util
from factored_gym import ActionTransform
from factored_gym.controller.controller import *


# class PLongitudinalController(object):
#
#     def __init__(self, process: Process, K_P=1.0, K_D=0.0, K_I=0.0):
#
#         self._vehicle = process.ego_vehicle
#         self._K_P = K_P
#         self._K_D = K_D
#         self._K_I = K_I
#         self._dt = process.time_step_length
#         self._e_buffer = deque(maxlen=30)
#
#     def run_step(self, target_speed, debug=True):
#         current_speed = np.sqrt(self._vehicle.internal_longitudinal_velocity ** 2 + \
#                                 self._vehicle.internal_lateral_velocity ** 2)
#
#         if debug:
#             print('Current speed = {}'.format(current_speed))
#
#         return self._pid_control(target_speed, current_speed)
#
#     def _pid_control(self, target_speed, current_speed):
#         """
#         Estimate the throttle of the vehicle based on the PID equations
#         :param target_speed:  target speed in Km/h
#         :param current_speed: current speed of the vehicle in Km/h
#         :return: throttle control in the range [0, 1]
#         """
#         _e = (target_speed - current_speed)
#         self._e_buffer.append(_e)
#
#         if len(self._e_buffer) >= 2:
#             _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
#             _ie = sum(self._e_buffer) * self._dt
#         else:
#             _de = 0.0
#             _ie = 0.0
#
#         return np.clip((self._K_P * _e) + (self._K_D * _de / self._dt) + \
#                        (self._K_I * _ie * self._dt), 0.0, 1.0)
from k_road.scan.waypoint_path_follower import WaypointPathFollower


class PLateralController(object):
    
    def __init__(self, process: Process, K_P=1.0):
        self._vehicle = process.ego_vehicle
        self._K_P = K_P
    
    def run_step(self, waypoint, debug=False):
        return self._p_control(waypoint)
    
    def _p_control(self, waypoint):
        position = self._vehicle.position
        vec_to_waypoint = waypoint - position
        error_angle = util.signed_delta_angle(vec_to_waypoint.angle, self._vehicle.angle)
        return error_angle * self._K_P


class VehicleProportionalController(ActionTransform, Controller):
    
    def __init__(self, args_lateral=None, args_longitudinal=None, look_a=None):
        
        if args_lateral is None:
            self.args_lateral = {'K_P': .05}
        else:
            self.args_lateral = args_lateral
        
        if args_longitudinal is None:
            self.args_longitudinal = {'K_P': 0.0}
        else:
            self.args_longitudinal = args_longitudinal
        
        if look_a is None:
            self.ld = 5
        else:
            self.ld = look_a
    
    def get_action(self, process: Process):
        return self.transform_action(process, np.array([0.0, 0.0]))
    
    def transform_action(self, process: Process, action):
        
        # self._lon_controller = PIDLongitudinalController(process, \
        #                                                  **self.args_longitudinal)
        self._lat_controller = PLateralController(process, **self.args_lateral)
        
        waypoint_path_follower = WaypointPathFollower(
            process.space,
            process.cross_track.point,
            0.1,
            True
            )
        
        future_crosstrack = waypoint_path_follower.get_next_waypoint(self.ld)[1]
        
        # waypoint = np.array([process.cross_track.point.x, \
        #                                process.cross_track.point.y])
        
        # waypoint = np.array([future_crosstrack[0], future_crosstrack[1]])
        
        [throttle, steering] = self.run_step(waypoint=future_crosstrack)
        print("PID u_optimal: ", [throttle, steering])
        return np.array([throttle, steering])
    
    def run_step(self, waypoint, target_speed=25, debug=False):
        
        # throttle = self._lon_controller.run_step(target_speed, debug)
        throttle = 0.0
        steering = self._lat_controller.run_step(waypoint, debug)
        
        return [throttle, steering]
