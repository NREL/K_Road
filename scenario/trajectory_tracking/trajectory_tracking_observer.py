from math import (
    cos,
    sin,
    )

import numpy as np
from gym import spaces
from pymunk import Vec2d

from k_road.constants import Constants
from k_road.k_road_view import KRoadView
from k_road.scan.waypoint_path_follower import WaypointPathFollower
from k_road.util import *
from factored_gym import (
    Observer,
    Optional,
    )
from factored_gym.action_transform.additive_action_transform import AdditiveActionTransform
from scenario.trajectory_tracking.trajectory_tracking_process import TrajectoryTrackingProcess


class TrajectoryTrackingObserver(Observer):
    """
        Objective:
            + Move along a planned path at a desired speed
            + Error factors:
                + deviation from desired speed
                + deviation from desired path
                + future: energy consumption
        
        Possible observation values:
            + heading error relative to points along the path (stanley)
            + cross track error at given point:
                + at some distance projected ahead of the rear axel at the heading angle (pure pursuit)
                + at front axel (stanley)
                + at rear axel
            + waypoints along path:
                + waypoint heading
                + waypoint distance
                + waypoint velocity (?)
                + waypoint yaw (?)

        - test:
            - waypoint spacing
                - scaling factor  *
                - based on: *
                    - vehicle velocity
                    - target velocity
            - waypoint choice:
                - cross track point
                - cross tracks projected in front of vehicle
            - waypoint angle:
                - relative to yaw *
                - relative to velocity angle *
                - relative to previous waypoint
                - combinations *
            - waypoint distances?
    """
    
    def __init__(
            self,
            scaling: float = 1.0,
            additive_transform: Optional[AdditiveActionTransform] = None,
            use_controller_action: bool = False,
            num_waypoints: int = 10,
            waypoint_spacing: float = 1.0,
            use_angles: bool = True,
            use_distances: bool = True,
            use_relative_distances: bool = False,
            use_target_speeds: bool = False,
            use_velocity_reference_angle: bool = False,
            use_alternate_reference_angle: bool = False,
            use_relative_angles: bool = False,
            space_waypoints_with_actual_speed=False,
            mirror: bool = False,
            mirror_using_far_waypoint: bool = False,
            use_cos: bool = False,
            use_sin: bool = False,
            ) -> None:
        self.scaling: float = scaling
        self.additive_transform: Optional[AdditiveActionTransform] = additive_transform
        self.use_controller_action: bool = use_controller_action
        self.num_waypoints: int = num_waypoints
        self.waypoint_spacing: float = waypoint_spacing
        self.use_angles: bool = use_angles
        self.use_distances: bool = use_distances
        self.use_relative_distances: bool = use_relative_distances
        self.use_target_speeds: bool = use_target_speeds
        self.use_velocity_reference_angle: bool = use_velocity_reference_angle
        self.use_alternate_reference_angle: bool = use_alternate_reference_angle
        self.use_relative_angles: bool = use_relative_angles
        self.space_waypoints_with_actual_speed: bool = space_waypoints_with_actual_speed
        self.mirror: bool = mirror
        self.mirror_using_far_waypoint: bool = mirror_using_far_waypoint
        self.use_cos: bool = use_cos
        self.use_sin: bool = use_sin
        
        # if self.mirror and self.additive_transform is None:
        #     assert False
        
        # self.time_limit: float = time_limit
        
        self.observation_length: int = 0
        
        self.controller_action_index: int = self.observation_length
        if self.use_controller_action:
            self.observation_length += 2
        
        self.target_speed_index: int = self.observation_length
        self.observation_length += 1
        
        # self.target_speed_error_index: int = self.observation_length
        # self.observation_length += 1
        
        # self.time_remaining_index: int = self.observation_length
        # self.observation_length += 1
        #
        # self.distance_from_current_waypoint_index: int = self.observation_length
        # self.observation_length += 1
        
        # self.relative_yaw_heading_index = self.observation_length
        # self.observation_length += 1
        
        self.yaw_rate_index: int = self.observation_length
        self.observation_length += 1
        
        self.steer_angle_index: int = self.observation_length
        self.observation_length: int = self.observation_length + 1
        #
        # self.acceleration_index  : int =  self.observation_length
        # self.observation_length  : int =  self.observation_length + 1
        
        self.lateral_velocity_index: int = self.observation_length
        self.observation_length += 1
        
        self.longitudinal_velocity_index: int = self.observation_length
        self.observation_length += 1
        
        # self.heading_error_sin_index: int = self.observation_length
        # self.observation_length += 1
        #
        # self.heading_error_cos_index: int = self.observation_length
        # self.observation_length += 1
        #
        # self.rear_cross_track_error_index: int = self.observation_length
        # self.observation_length += 1
        #
        # self.front_cross_track_error_index: int = self.observation_length
        # self.observation_length += 1
        
        # self.cross_track_error_index: int = self.observation_length
        # self.observation_length += 1
        
        self.waypoint_info_index: int = self.observation_length
        
        self.num_values_per_waypoint: int = 0
        
        if self.use_angles:
            self.num_values_per_waypoint += 1
        if self.use_alternate_reference_angle:
            self.num_values_per_waypoint += 1
        if self.use_distances:
            self.num_values_per_waypoint += 1
        if self.use_relative_distances:
            self.num_values_per_waypoint += 1
        if self.use_target_speeds:
            self.num_values_per_waypoint += 1
        if self.use_relative_angles:
            self.num_values_per_waypoint += 1
        if self.use_cos:
            self.num_values_per_waypoint += 1
        if self.use_sin:
            self.num_values_per_waypoint += 1
        
        self.observation_length += self.num_waypoints * self.num_values_per_waypoint
        # self.waypoint_path_follower = None
        
        # self.distance_to_target_index: int = self.observation_length
        # self.observation_length += 1
        
        # self.time_left_index: int = self.observation_length
        # self.observation_length += 1
        
        # self.baseline_acceleration_index  : int =  self.observation_length
        # self.observation_length  : int =  self.observation_length + 1
        #
        # self.baseline_steer_angle_index  : int =  self.observation_length
        # self.observation_length  : int =  self.observation_length + 1
        
        self.observation_space = \
            spaces.Box(low=-self.scaling, high=self.scaling, shape=(self.observation_length,))
        
        self.reset(None)
    
    def reset(self, process: TrajectoryTrackingProcess) -> None:
        self.render_position = Vec2d(0, 0)
        self.render_yaw = 0
        self.render_velocity = Vec2d(0, 0)
        self.render_waypoints = []
        self.mirrored_waypoints = []
        if self.mirror:
            self.additive_transform.base_coefficient = 1.0
    
    def render(self, process: TrajectoryTrackingProcess, view: KRoadView) -> None:
        for waypoint in self.render_waypoints:
            view.draw_circle((0, 192, 0), waypoint[0], .6)
        # if process.mirror_steering < 0:
        #     for waypoint in self.mirrored_waypoints:
        #         view.draw_circle((0, 192, 192), waypoint, .5)
        
        ego = process.ego_vehicle
        position = ego.position
        view.draw_line((192, 0, 192), position,
                       Vec2d(ego.internal_longitudinal_velocity * 1.0 + 10, 0).rotated(ego.angle) + position)
        view.draw_line((55, 204, 189),
                       position,
                       Vec2d(ego.internal_longitudinal_velocity, ego.internal_lateral_velocity).rotated(
                           ego.angle) * 1.5 + position)
    
    def get_observation_space(self, process: TrajectoryTrackingProcess):
        return self.observation_space
    
    def get_observation(self, process: TrajectoryTrackingProcess):
        # print('get_observation')
        # printing = random.randrange(4) == 0
        # printing = True
        printing = False
        
        # compute observation
        observation = np.zeros(self.observation_space.shape)
        
        # time_step_number: int = process.time_step_number
        ego = process.ego_vehicle
        yaw = ego.angle
        velocity = Vec2d(ego.internal_longitudinal_velocity, ego.internal_lateral_velocity).rotated(yaw)
        velocity_heading = velocity.angle if velocity.length > .1 else yaw
        
        reference_angle = velocity_heading if self.use_velocity_reference_angle else yaw
        alternate_reference_angle = yaw if self.use_velocity_reference_angle else velocity_heading
        
        cross_track = process.cross_track
        reference_position = process.position
        
        waypoint_angle = reference_angle
        
        self.render_waypoints.clear()
        self.mirrored_waypoints.clear()
        waypoints = self.render_waypoints
        
        target_speed = process.target_speed
        if cross_track is not None:
            waypoint_path_follower = WaypointPathFollower(
                process.space,
                cross_track.point,
                process.max_scan_radius,
                start_mid_path=True,
                path=cross_track.shape.body.entity,
                )
            
            waypoint_delta = 0.0
            waypoint_target_speed = target_speed
            for i in range(self.num_waypoints):
                
                if self.space_waypoints_with_actual_speed:
                    waypoint_delta = float(
                        ego.internal_longitudinal_velocity * process.time_step_length * self.waypoint_spacing)
                else:
                    waypoint_delta = float(
                        waypoint_target_speed * process.time_step_length * self.waypoint_spacing)
                
                state, waypoint = waypoint_path_follower.get_next_waypoint(waypoint_delta)
                
                if waypoint_path_follower.path is not None:
                    waypoint_target_speed = waypoint_path_follower.path.target_speed
                
                waypoints.append((waypoint, waypoint_target_speed))
            waypoint_angle = (waypoints[-1][0] - reference_position).angle \
                if self.mirror_using_far_waypoint else \
                (waypoints[0][0] - reference_position).angle
        
        cross_track_delta_angle = signed_delta_angle(waypoint_angle, reference_angle)
        
        mirror_coefficient = -1.0 if cross_track_delta_angle < 0.0 and self.mirror else 1.0
        
        if self.mirror:
            self.additive_transform.base_coefficient = mirror_coefficient
        # process.mirror_steering = mirror_coefficient
        
        if self.use_controller_action:
            action = self.additive_transform.input.get_action(process)
            controller_acceleration = hinge_transform(
                action[0], 1.0 / Constants.max_deceleration, 1.0 / Constants.max_acceleration)
            controller_steering = action[1] / Constants.max_steer_angle
            observation[self.controller_action_index] = clamp(controller_acceleration)
            observation[self.controller_action_index + 1] = clamp(controller_steering)
        
        observation[self.target_speed_index] = scale_and_clamp(target_speed, 45)
        
        # observation[self.relative_yaw_heading_index] = \
        #     mirror_coefficient * scale_and_clamp(delta_angle(alternate_reference_angle, reference_angle), pi)
        
        observation[self.yaw_rate_index] = \
            mirror_coefficient * scale_and_clamp(ego.internal_angular_velocity, pi)  # 180 deg / sec cap
        
        observation[self.steer_angle_index] = \
            scale_and_clamp(ego.steer_angle * mirror_coefficient, Constants.max_steer_angle)
        
        # observation[self.acceleration_index] = ego.acceleration / Constants.max_acceleration
        
        observation[self.lateral_velocity_index] = \
            mirror_coefficient * scale_and_clamp(ego.internal_lateral_velocity, 10)
        
        observation[self.longitudinal_velocity_index] = \
            scale_and_clamp(ego.internal_longitudinal_velocity, Constants.max_speed)
        
        waypoint_distance_scale = Constants.max_speed * process.time_step_length * self.waypoint_spacing
        previous_waypoint = reference_position  # reference_position
        
        for i, wp in enumerate(waypoints):
            waypoint, waypoint_target_speed = wp
            
            relative_delta = waypoint - previous_waypoint
            relative_angle = mirror_coefficient * relative_delta.angle
            
            delta = waypoint - reference_position  # reference_position
            angle = delta.angle
            
            relative_delta_angle = signed_delta_angle(angle, reference_angle)
            # mirrored_waypoint = reference_position + Vec2d(delta.length, 0).rotated(reference_angle).rotated(
            # -relative_delta_angle)
            mirrored_waypoint = reference_position + Vec2d(delta.length, 0).rotated(reference_angle).rotated(
                -relative_delta_angle)
            # mirrored_waypoint = reference_position + Vec2d(delta.length, 0).rotated(reference_angle).rotated(
            # relative_delta_angle)
            # mirrored_waypoint = reference_position + Vec2d(delta.length, 0).rotated(angle)
            # delta_angle(reference_angle,  delta_angle(angle, reference_angle)))
            self.mirrored_waypoints.append(mirrored_waypoint)
            
            heading_error = mirror_coefficient * relative_delta_angle
            
            if printing:
                print('data: ', i, waypoint_target_speed,
                      'actual:', delta, delta.length, angle, signed_delta_angle(angle, reference_angle),
                      'relative:', relative_delta, relative_delta.length, relative_angle)
            
            base = self.waypoint_info_index + i * self.num_values_per_waypoint
            offset = 0
            
            if self.use_angles:  # default
                observation[base + offset] = scale_and_clamp(heading_error, pi)
                offset += 1
            
            if self.use_alternate_reference_angle:
                alternate_angle = mirror_coefficient * signed_delta_angle(angle, alternate_reference_angle)
                observation[base + offset] = scale_and_clamp(alternate_angle, pi)
                offset += 1
            
            if self.use_relative_angles:
                observation[base + offset] = scale_and_clamp(relative_angle, pi)
                offset += 1
            
            if self.use_distances:  # default
                observation[base + offset] = scale_and_clamp(
                    delta.length,
                    waypoint_distance_scale * (i + 1))
                offset += 1
            
            if self.use_relative_distances:
                observation[base + offset] = scale_and_clamp(relative_delta.length, waypoint_distance_scale)
                offset += 1
            
            if self.use_target_speeds:
                observation[base + offset] = scale_and_clamp(waypoint_target_speed, Constants.max_speed)
                offset += 1
            
            if self.use_cos:
                observation[base + offset] = cos(heading_error)
                offset += 1
            
            if self.use_sin:
                observation[base + offset] = sin(heading_error)
                offset += 1
            
            previous_waypoint = waypoint
        
        observation = np.multiply(self.scaling, observation)
        
        if printing:
            print('obs: ', np.transpose(observation))
        return observation
