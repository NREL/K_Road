import random

from pymunk import Vec2d

from k_road.constants import Constants
from factored_gym.rewarder import Rewarder
from scenario.trajectory_tracking.trajectory_tracking_process import TrajectoryTrackingProcess


class TrajectoryTrackingRewarder(Rewarder):
    """
    Objective
    """
    
    def __init__(
            self,
            speed_tradeoff: float = .0025,
            steering_control_effort_tradeoff: float = .0016,
            acceleration_control_effort_tradeoff: float = .00027,
            max_steering_control_effort: float = 2 * Constants.max_steer_angle,
            ) -> None:
        
        self.speed_tradeoff: float = speed_tradeoff
        self.steering_control_effort_tradeoff: float = steering_control_effort_tradeoff
        self.max_steering_control_effort: float = max_steering_control_effort
        self.acceleration_control_effort_tradeoff: float = acceleration_control_effort_tradeoff
        
        self.previous_steer_angle: float = 0.0
        self.previous_acceleration: float = 0.0
        self.maximum_penalty_per_meter: float = 0.0
    
    def reset(self, process: TrajectoryTrackingProcess) -> None:
        self.previous_steer_angle = process.ego_vehicle.steer_angle
        self.previous_acceleration = process.ego_vehicle.acceleration
        
        self.maximum_penalty_per_meter = (process.max_cross_track_error ** 2 +
                                          self.speed_tradeoff * process.max_speed_error ** 2 +
                                          self.steering_control_effort_tradeoff *
                                          (self.max_steering_control_effort / process.time_step_length) ** 2 +
                                          self.acceleration_control_effort_tradeoff * (
                                                  Constants.max_acceleration + Constants.max_deceleration) /
                                          process.time_step_length)
    
    def get_reward(self, process: TrajectoryTrackingProcess, observation, terminated: bool) -> float:
        # positions, cross_tracks, cross_track_positions = self.get_positions_and_cross_track(process)
        ego_vehicle = process.ego_vehicle
        
        cross_track = process.cross_track
        previous_cross_track = process.previous_cross_track
        
        reward = 0.0
        
        if cross_track is not None and previous_cross_track is not None:
            path_delta: Vec2d = cross_track.point - previous_cross_track.point
            
            previous_path = cross_track.shape.body.entity
            path = previous_cross_track.shape.body.entity
            
            path_direction = previous_path.direction + path.direction
            # path_distance_traveled = copysign(path_delta.length, path_direction.dot(path_delta))
            path_distance_traveled = path_delta.length
            
            # --- find integrated squared cross track error
            def project_to_path_reference(pos: Vec2d) -> Vec2d:
                return (pos - previous_cross_track.point).rotated(-path_direction.angle)
            
            p0 = project_to_path_reference(process.previous_position)
            p1 = project_to_path_reference(process.position)
            
            squared_cte = self.get_integrated_squared_error(p0.y, p1.y)
            
            # --- find integrated squared speed error
            # speed_error = path.target_speed - ego_vehicle.longitudinal_velocity
            speed_error = path.target_speed - ego_vehicle.instantaneous_body_velocity.length
            squared_speed_error = speed_error ** 2
            
            # --- find squared steering control effort
            steering_control_effort = \
                ((ego_vehicle.steer_angle - self.previous_steer_angle) / process.time_step_length) ** 2
            
            # --- find squared acceleration control effort
            acceleration_control_effort = \
                ((ego_vehicle.acceleration - self.previous_acceleration) / process.time_step_length) ** 2
            
            # --- compute total penalty for this time-step
            reward -= path_distance_traveled * \
                      (squared_cte +
                       self.speed_tradeoff * squared_speed_error +
                       self.steering_control_effort_tradeoff * steering_control_effort +
                       self.acceleration_control_effort_tradeoff * acceleration_control_effort)
            
            # --- if we ended without reaching the end of the path, add a penalty equal to the max penalty remaining
            if terminated and not process.reached_end:
                path_length_remaining = process.path_length_remaining
                reward -= path_length_remaining * self.maximum_penalty_per_meter
                # print('terminated reward: ', reward, path_length_remaining)
                
                if random.randrange(400) == 0:
                    # if random.randrange(1) == 0:
                    # if True:
                    print('reward', reward,
                          'squared_cte', squared_cte,
                          'squared_speed_error', squared_speed_error,
                          'steering_control_effort', steering_control_effort,
                          'path_distance_traveled', path_distance_traveled)
        
        self.previous_steer_angle = ego_vehicle.steer_angle
        self.previous_acceleration = ego_vehicle.acceleration
        
        return reward
    
    @staticmethod
    def get_integrated_squared_error(
            previous_error: float,
            current_error: float,
            duration: float = 1.0,
            ) -> float:
        return (duration / 3.0) * \
               (previous_error * previous_error + previous_error * current_error + current_error * current_error)
    
    # def get_reward(self, process: TrajectoryTrackingProcess, observation, terminated: bool) -> float:
    #     # positions, cross_tracks, cross_track_positions = self.get_positions_and_cross_track(process)
    #     ego_vehicle = process.ego_vehicle
    #     positions = ego_vehicle.positions
    #
    #     cross_track = process.cross_track
    #     previous_cross_track = process.previous_cross_track
    #
    #     reward = 0.0
    #
    #     if cross_track is not None and previous_cross_track is not None:
    #         path_delta: Vec2d = cross_track.point - previous_cross_track.point
    #
    #         previous_path = cross_track.shape.body.entity
    #         path = previous_cross_track.shape.body.entity
    #
    #         path_direction = previous_path.direction + path.direction
    #         path_distance_traveled = copysign(path_delta.length, path_direction.dot(path_delta))
    #
    #         # --- find integrated squared cross track error
    #         def project_to_path_reference(pos: Vec2d) -> Vec2d:
    #             return (pos - previous_cross_track.point).rotated(-path_direction.angle)
    #
    #         p0 = project_to_path_reference(process.previous_position)
    #         p1 = project_to_path_reference(process.position)
    #
    #         squared_cte = self.get_integrated_squared_error(p0.y, p1.y)
    #
    #         # --- find integrated squared speed error
    #         speed_error = previous_path.target_speed + path.target_speed) - ego_vehicle.longitudinal_velocity
    #         squared_speed_error = self.get_integrated_squared_error(self.previous_speed_error, speed_error)
    #         self.previous_speed_error = speed_error
    #
    #         # --- find squared steering control effort
    #         steering_control_effort = \
    #             ((ego_vehicle.steer_angle - self.previous_steer_angle) / process.time_step_length) ** 2
    #
    #         # --- compute total penalty for this time-step
    #         reward += -path_distance_traveled * \
    #                   (squared_cte +
    #                    self.speed_tradeoff * squared_speed_error +
    #                    self.steering_control_effort_tradeoff * steering_control_effort)
    #
    #     # --- if we ended without reaching the end of the path, add a penalty equal to the max penalty remaining
    #     if terminated and not process.reached_end:
    #         path_length_remaining = process.path_length_remaining
    #         reward -= path_length_remaining * self.maximum_penalty_per_meter
    #         print('terminated reward: ', reward, path_length_remaining)
    #
    #     if random.randrange(400) == 0:
    #         # if random.randrange(1) == 0:
    #         # if True:
    #         print('reward', reward,
    #               'squared_cte', squared_cte,
    #               'squared_speed_error', squared_speed_error,
    #               'steering_control_effort', steering_control_effort,
    #               'path_distance_traveled', path_distance_traveled)
    #
    #     self.previous_speed_error = speed_error
    #     self.previous_positions = positions
    #     self.previous_cross_tracks = cross_tracks
    #     self.previous_cross_track_positions = cross_track_positions
    #     self.previous_steer_angle = ego_vehicle.steer_angle
    #     return reward
    
    # @staticmethod
    # def get_positions_and_cross_track(process: TrajectoryTrackingProcess):
    #     ego_vehicle = process.ego_vehicle
    #     positions = ego_vehicle.positions
    #     cross_tracks = process.cross_tracks
    #     cross_track_positions = [p.point if p is not None else positions[i] for i, p in
    #                              enumerate(cross_tracks)]
    #     return positions, cross_tracks, cross_track_positions
    
    # def get_integrated_cte(
    #         self,
    #         previous_cross_track_position: Vec2d,
    #         previous_position: Vec2d,
    #         cross_track_position: Vec2d,
    #         position: Vec2d) -> float:
    #     path_delta: Vec2d = cross_track_position - previous_cross_track_position
    #     reference_position: Vec2d = previous_cross_track_position
    #     reference_angle: float = -path_delta.angle
    #
    #     def project_to_path_reference(pos: Vec2d) -> Vec2d:
    #         return (pos - reference_position).rotated(reference_angle)
    #
    #     x0: float = project_to_path_reference(previous_position).y
    #     x1: float = project_to_path_reference(position).y
    #     # reflect x's so that x0 >= 0
    #     if x0 < 0:
    #         x0 = -x0
    #         x1 = -x1
    #
    #     cross_track_error: float = 0
    #     if x1 >= 0:
    #         cross_track_error = .5 * (x0 + x1)
    #     else:
    #         c: float = x0 / (x0 - x1)  # find intersection point
    #         cross_track_error = .5 * (c * x0 + (1 - c) * (-x1))
    #     return cross_track_error
