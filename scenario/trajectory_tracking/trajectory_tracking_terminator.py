import random
from math import *

from factored_gym import Terminator
from scenario.trajectory_tracking.trajectory_tracking_process import TrajectoryTrackingProcess


class TrajectoryTrackingTerminator(Terminator):

    def __init__(self) -> None:
        pass

    def is_terminal(self, process: TrajectoryTrackingProcess, observation) -> bool:
        # print('is_terminal')
        ego_vehicle = process.ego_vehicle
        position = ego_vehicle.position
        cross_track = process.cross_track
        previous_cross_track = process.previous_cross_track

        wrong_way = False
        wrong_heading = False
        path_distance_traveled = 0.0
        if cross_track is not None and previous_cross_track is not None:
            path_delta = (cross_track.point - previous_cross_track.point)

            path = cross_track.shape.body.entity
            previous_path = previous_cross_track.shape.body.entity

            path_direction = previous_path.direction + path.direction
            path_distance_traveled = copysign(path_delta.length, path_direction.dot(path_delta))

            if previous_path is path:
                wrong_way = path_distance_traveled <= .1 * process.time_step_length
                # if wrong_way:
                #     print(path_distance_traveled, path_direction, path_delta.length, path_direction.dot(path_delta),
                #           path_direction.dot(path_delta.normalized()))
            else:
                wrong_way = previous_path.sequence_id > path.sequence_id
                # if wrong_way:
                #     print(previous_path.sequence_id, path.sequence_id)
            # if random.randrange(10) == 0:
            #     print('pdt', path_distance_traveled)
            # heading_error = delta_angle(ego_vehicle.angle, path_delta.angle)
            # wrong_heading = fabs(heading_error) > (60.0 * pi / 180.0)

        too_far_from_waypoint = \
            cross_track is None or \
            (position - cross_track.point).length > process.max_cross_track_error

        exceeded_max_speed_error = \
            fabs(ego_vehicle.internal_longitudinal_velocity - process.target_speed) > process.max_speed_error

        vehicle_model_out_of_bounds = not ego_vehicle.is_in_valid_state()

        terminal = process.reached_end or \
                   wrong_way or \
                   wrong_heading or \
                   too_far_from_waypoint or \
                   exceeded_max_speed_error or \
                   vehicle_model_out_of_bounds

        if terminal and random.randrange(500) == 0:
        # if terminal:
            print('term:',
                  'end', process.reached_end,
                  'model', vehicle_model_out_of_bounds,
                  'wrong way', wrong_way,
                  'wrong heading', wrong_heading,
                  # 'progress', no_progress,
                  'too far', too_far_from_waypoint,
                  'speed error', exceeded_max_speed_error,
                  # delta.dot(last_segment),
                  process.cross_track.distance,
                  (ego_vehicle.internal_longitudinal_velocity - process.target_speed),
                  (ego_vehicle.velocity.length - process.target_speed),
                  ego_vehicle.internal_longitudinal_velocity,
                  ego_vehicle.internal_lateral_velocity,
                  process.target_speed,
                  path_distance_traveled,
                  process.path_length_remaining)

        return terminal
