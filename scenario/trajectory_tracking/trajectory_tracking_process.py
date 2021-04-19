from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple,
    )

import pymunk
from pymunk import (
    PointQueryInfo,
    Vec2d,
    )

from k_road.builder.road_builder import RoadBuilder
from k_road.constants import Constants
from k_road.entity.entity_type import EntityType
from k_road.entity.path import Path
from k_road.entity.vehicle.dynamic_single_body_vehicle import DynamicSingleBodyVehicle
from k_road.entity.vehicle.vehicle_dbm3 import VehicleDBM3
from k_road.k_road_view import KRoadView
from k_road.scan import pathfinder
from k_road.single_vehicle_process import SingleVehicleProcess
from k_road.util import *
from factored_gym import Process
from scenario.trajectory_tracking.trajectory_generators import curriculum_trajectory_generator


class TrajectoryTrackingProcess(SingleVehicleProcess):
    reached_end: bool
    metadata = {
        'render.modes': ['human']
        }
    
    def __init__(self,
                 path_generator: Optional[
                     Callable[['TrajectoryTrackingProcess'], Tuple[List[Tuple[Vec2d, float]], bool]]] = None,
                 view_distance: float = 100.0,
                 # view_distance=1000.0,
                 scale: float = 16.0,
                 # scale=1.0,
                 get_starting_point: Optional[
                     Callable[['TrajectoryTrackingProcess'], Tuple[Vec2d, Vec2d, float, float]]] = None,
                 make_ego_vehicle: Optional[
                     Callable[['TrajectoryTrackingProcess', Vec2d, Vec2d, float, float], DynamicSingleBodyVehicle]] = None,
                 # max_scan_radius: float = 50.0,
                 differential_steering: bool = False,
                 cross_track_position: int = 0,
                 max_cross_track_error: float = 5.0,
                 max_speed_error: float = Constants.max_speed,
                 end_buffer_distance: float = 100,
                 scale_actions: bool = False,
                 **kwargs
                 ):
        # print('init 0')
        super().__init__(view_follows_ego=True, **kwargs)
        # print('init 1')
        self.scale: float = scale
        
        self.path_generator: Callable[['TrajectoryTrackingProcess'], Tuple[List[Tuple[Vec2d, float]], bool]] = \
            curriculum_trajectory_generator if path_generator is None else path_generator
        
        self.get_starting_point: Callable[['TrajectoryTrackingProcess'], Tuple[Vec2d, Vec2d, float, float]] = \
            self.default_get_starting_point if get_starting_point is None else get_starting_point
        
        self.make_ego_vehicle: \
            Callable[['TrajectoryTrackingProcess', Vec2d, Vec2d, float, float], DynamicSingleBodyVehicle] = \
            self.default_make_ego_vehicle if make_ego_vehicle is None else make_ego_vehicle
        
        self.max_scan_radius: float = max_cross_track_error + 10
        self.differential_steering: bool = differential_steering
        self.cross_track_position: int = cross_track_position
        self.max_cross_track_error: float = max_cross_track_error
        self.max_speed_error: float = max_speed_error
        self.end_buffer_distance: float = end_buffer_distance
        self.scale_actions: bool = scale_actions
        
        self.view_distance: float = view_distance
        self.road: Optional[RoadBuilder] = None
        
        self.is_cyclical: bool = False

        # print('init 2')
        self._initialize_summary_vars()
        # print('init 3')
    
    def make_view(self, mode) -> KRoadView:
        # print('make_view')
        padding = .01
        size = 1.0 * self.view_distance * (1.0 + padding)
        view_size = int(ceil(self.scale * size))

        return KRoadView(
            (view_size, view_size),
            self.ego_vehicle.position,
            self.scale,
            self.time_dilation)
    
    def reset(self, process: Process) -> None:
        # print('reset')
        
        
        for entity in list(self.entities):
            entity.discard()

        super().reset(self)
        
        self._initialize_summary_vars()
        
        waypoints, self.is_cyclical = self.path_generator(self)
        
        if not self.is_cyclical:
            def make_buffered_waypoint(first, second):
                return first[0] + (first[0] - second[0]).normalized() * self.end_buffer_distance, first[1]
            waypoints = waypoints + [make_buffered_waypoint(waypoints[-1], waypoints[-2])]
        
        positions = [wp[0] for wp in waypoints]
        
        if self.is_cyclical:
            self.road = RoadBuilder(
                self,
                positions[-1],
                positions + [positions[0]],
                positions[1],
                [EntityType.path])
        else:
            self.road = RoadBuilder(
                self,
                None,
                positions,
                None,
                [EntityType.path])
        
        for i, path in enumerate(self.road.entities[0]):
            path.target_speed = waypoints[i][1]
        
        
        position, velocity, yaw, angular_velocity = self.get_starting_point(self)
        self.ego_vehicle = self.make_ego_vehicle(self, position, velocity, yaw, angular_velocity)
        self._update_summary_vars()
    
    @property
    def target_speed(self) -> float:
        cross_track = self.cross_track
        if cross_track is None:
            print('cross track is none on target speed!')
            return 10.0
        return cross_track.shape.body.entity.target_speed
    
    @property
    def path_length_remaining(self) -> float:
        path_pos = self.road.global_to_road(self.ego_vehicle.front_position)
        return max(0.0, self.road.length - path_pos.x - self.end_buffer_distance)
    
    def step(self, action) -> Any:
        # print('step', action)
        if self.differential_steering:
            action[1] = action[1] + self.ego_vehicle.steer_angle
        
        if self.scale_actions:
            self.ego_vehicle.apply_normalized_control_action(self.time_step_length, action)
        else:
            # print('apply ', action[0], action[1])
            self.ego_vehicle.apply_control(self.time_step_length,
                                           clamp(action[0], Constants.max_deceleration, Constants.max_acceleration),
                                           clamp(action[1], -Constants.max_steer_angle, Constants.max_steer_angle))
        
        super().step(action)
        self._update_summary_vars()
        
        reached_end = False
        cross_track = self.cross_track
        if not self.is_cyclical and cross_track is not None and \
                cross_track.shape.body.entity is self.road.entities[0][-1]:
            reached_end = True
        self.reached_end = reached_end

        # print('step done')
        return {}
    
    def render(self, process: 'KRoadProcess', view: KRoadView) -> None:
        # print('render')
        super().render(process, view)
        for p in self.ego_vehicle.positions:
            view.draw_circle((200, 0, 64), p, .2)
        
        if self.cross_track is not None:
            view.draw_circle((64, 0, 200), self.cross_track.point, .3)
    
    def get_cross_track(self,
                        position: Vec2d,
                        previous_path: Optional[Path] = None,
                        ) -> pymunk.PointQueryInfo:
        
        cross_track, path = pathfinder.find_best_path_hint(
            self.space, position, self.max_scan_radius, previous_path=previous_path)
        if cross_track is None:
            print('cross track is none! ', position, self.max_scan_radius, previous_path)
        return cross_track
    
    def _initialize_summary_vars(self) -> None:
        # self.cross_tracks: [Optional[PointQueryInfo]] = [None] * 3
        # self.previous_cross_tracks: [Optional[PointQueryInfo]] = [None] * 3
        self.cross_track: Optional[PointQueryInfo] = None
        self.previous_cross_track: Optional[PointQueryInfo] = None
        self.reached_end: bool = False
    
    def _update_summary_vars(self) -> None:
        self.previous_positions = self.ego_vehicle.positions
        self.previous_cross_track = self.cross_track
        
        ct = self.cross_track
        previous_path = None if ct is None else ct.shape.body.entity
        
        self.cross_track = self.get_cross_track(self.ego_vehicle.positions[0], previous_path=previous_path)
    
    @property
    def position(self) -> Vec2d:
        return self.ego_vehicle.positions[self.cross_track_position]
    
    @property
    def previous_position(self) -> Vec2d:
        return self.previous_positions[self.cross_track_position]
    
    @property
    def path_segments(self) -> [Path]:
        return self.road.entities[0]
    
    @property
    def end_path(self) -> Path:
        return self.path_segments[-1]
    
    @staticmethod
    def default_make_ego_vehicle(
            process: 'TrajectoryTrackingProcess',
            position: Vec2d,
            velocity: Vec2d,
            yaw: float,
            angular_velocity: float
            ) -> DynamicSingleBodyVehicle:
        return VehicleDBM3(
            process,
            (255, 64, 64),
            position,
            velocity,
            yaw=yaw,
            angular_velocity=angular_velocity
            )
    
    @staticmethod
    def default_get_starting_point(process: 'TrajectoryTrackingProcess') -> (Vec2d, Vec2d, float, float):
        initial_path: Path = process.path_segments[0]
        # cross_track_error = 0.0 * (2 * random.random() - 1)
        cross_track_error = 0
        # heading_error = pi * (0.0 / 180.0) * (2 * random.random() - 1)
        heading_error = 0
        position = initial_path.start + initial_path.perpendicular * cross_track_error
        yaw = reset_angle(initial_path.angle + heading_error)
        target_speed = initial_path.target_speed
        velocity = Vec2d(target_speed, 0)
        angular_velocity = 0
        return position, velocity, yaw, angular_velocity
