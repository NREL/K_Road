import math
from typing import Optional

import numpy as np
import pymunk
from gym import spaces
from pymunk import Vec2d

from factored_gym import Observer
from k_road.constants import Constants
from k_road.entity.entity_type import EntityType
from k_road.k_road_view import KRoadView
from k_road.scan import pathfinder
from k_road.util import *
from scenario.road import RoadProcess
from scenario.road.ray_scanner import RayScanner


class RoadObserver(Observer):

    def __init__(self,
                 scaling: float = 1.0,
                 forward_scan_angle=(30.0 / 360) * 2 * math.pi,
                 forward_scan_resolution: int = 33,  # 33,
                 forward_scan_radius: float = .1,  # 3.7 / 2.0,
                 forward_scan_distance: float = 200,  # 150-200m industry standard forward scan distance
                 rear_scan_resolution: int = 13,  # 33,
                 rear_scan_radius: float = .1,  # 3.7 / 2.0,
                 rear_scan_distance: float = 30,
                 ):
        self.scaling: float = scaling
        self.forward_scan_angle: float = forward_scan_angle
        self.forward_scan_resolution: int = forward_scan_resolution
        self.rear_scan_resolution: int = rear_scan_resolution

        self.observation_length: int = 0

        # 0
        self.distance_to_target_index: int = self.observation_length
        self.observation_length += 1

        # 1
        self.speed_along_path_index: int = self.observation_length
        self.observation_length += 1

        # 2
        self.heading_along_path_index: int = self.observation_length
        self.observation_length += 1

        # 3
        self.cross_track_error_index: int = self.observation_length
        self.observation_length += 1

        # 4
        self.yaw_rate_index: int = self.observation_length
        self.observation_length += 1

        # 5
        self.steer_angle_index: int = self.observation_length
        self.observation_length += 1

        # 6
        self.acceleration_index: int = self.observation_length
        self.observation_length += 1

        # 7
        self.lateral_velocity_index: int = self.observation_length
        self.observation_length += 1

        # 8
        self.longitudinal_velocity_index: int = self.observation_length
        self.observation_length += 1

        # self.time_left_index: int = self.observation_length
        # self.observation_length += 1

        # self.left_lane_marking_yellow_distance_index: int = self.observation_length
        # self.observation_length += 1
        #
        # self.right_lane_marking_yellow_distance_index: int = self.observation_length
        # self.observation_length += 1

        self.forward_scan_offset = self.forward_scan_angle / 2
        self.forward_scanner = \
            RayScanner(forward_scan_distance, self.forward_scan_angle, forward_scan_radius,
                       self.forward_scan_resolution)

        rear_scan_step = (2 * math.pi - self.forward_scan_angle) / (rear_scan_resolution + 2)
        rear_scan_angle = rear_scan_step * rear_scan_resolution
        self.rear_scan_offset = -self.forward_scan_offset + self.forward_scan_angle + rear_scan_step
        self.rear_scanner = \
            RayScanner(rear_scan_distance, rear_scan_angle, rear_scan_radius, self.rear_scan_resolution)

        # self.scan_resolution: int = 16
        # self.scan_distance: float = 30.0
        # self.scan_radius = .1
        # self.scan_arc: float = 2 * math.pi / self.scan_resolution

        # self.forward_scan_lane_marking_yellow_index: int = self.observation_length
        # self.observation_length: int = self.observation_length + self.forward_scan_resolution

        num_scan_arrays = 1

        self.forward_scan_vehicle_index: int = self.observation_length
        self.observation_length: int = self.observation_length + self.forward_scan_resolution * num_scan_arrays

        # self.forward_scan_vehicle_closing_index: int = self.observation_length
        # self.observation_length: int = self.observation_length + self.forward_scan_resolution

        # self.forward_scan_lane_marking_white_dashed_scan_index: int = self.observation_length
        # self.observation_length: int = self.observation_length + self.forward_scan_resolution

        # self.rear_scan_lane_marking_yellow_index: int = self.observation_length
        # self.observation_length: int = self.observation_length + self.rear_scan_resolution

        self.rear_scan_vehicle_index: int = self.observation_length
        self.observation_length: int = self.observation_length + self.rear_scan_resolution * num_scan_arrays

        # self.rear_scan_vehicle_closing_index: int = self.observation_length
        # self.observation_length: int = self.observation_length + self.rear_scan_resolution

        # self.rear_scan_lane_marking_white_dashed_scan_index: int = self.observation_length
        # self.observation_length: int = self.observation_length + self.rear_scan_resolution

        self.forward_scan_results = None
        self.rear_scan_results = None

        # self.baseline_acceleration_index = self.observation_length
        # self.observation_length += 1
        #
        # self.baseline_steer_angle_index = self.observation_length
        # self.observation_length += 1

        self.observation_space = \
            spaces.Box(low=-self.scaling, high=self.scaling, shape=(self.observation_length,))

        self.num = 0

    def get_observation_space(self, process: RoadProcess):
        return self.observation_space

    def reset(self, process: RoadProcess):
        # self.contacts.clear()
        # self.scan_endpoints.clear()
        # self.lane_marking_yellow_contacts.clear()
        # self.lane_contacts.clear()
        # self.scan_results = None

        # attach sensors to the ego vehicle
        # self.sensor_shapes = []
        # for scan_index in range(self.scan_resolution):
        #     scan_angle = -math.pi + self.scan_arc * scan_index
        #     vertices = [
        #         Vec2d(0, 0),
        #         Vec2d(self.scan_distance, 0),
        #         Vec2d(self.scan_distance * math.cos(self.scan_arc), self.scan_distance * math.sin(self.scan_arc))
        #     ]
        #
        #     vertices = [v.rotated(scan_angle) for v in vertices]
        #
        #     sensor_shape = pymunk.Poly(process.ego_vehicle.body, vertices)
        #     sensor_shape.sensor = True
        #     self.sensor_shapes.append(sensor_shape)
        #     sensor_shape.entity = process.ego_vehicle
        #
        # process.sensor_shapes = self.sensor_shapes
        pass

    def get_observation(self, process: RoadProcess):
        # compute observation
        observation = np.empty(self.observation_space.shape)

        ego_vehicle = process.ego_vehicle
        position = ego_vehicle.position
        velocity = ego_vehicle.velocity
        yaw = ego_vehicle.angle

        space = process.space

        # delta_angle = math.atan2(process.ego_to_target[1], process.ego_to_target[0])
        # heading = delta_angle(ego.angle, delta_angle)

        scan_position: Vec2d = position + Vec2d(ego_vehicle.length / 2, 0).rotated(yaw)
        path_pqi: Optional[pymunk.PointQueryInfo] = \
            pathfinder.find_best_path(space, scan_position, 1 * Constants.lane_width)[0]
        cross_track_error: float = 0 if path_pqi is None else (path_pqi.point - position).length
        best_path: Optional[Path] = None if path_pqi is None else path_pqi.shape.body.entity

        observation[self.distance_to_target_index] = \
            min(1.0, max(-1.0, 2 * (process.distance_to_target / process.target_offset) - 1))

        # maxed at 1
        speed_along_path: float = \
            velocity.dot(best_path.direction) if best_path is not None else 0
        observation[self.speed_along_path_index] = min(1.0, max(-1.0, speed_along_path / (1.1 * Constants.max_speed)))

        angle_agreement_with_path: float = \
            signed_delta_angle(best_path.direction.angle, velocity.angle) if best_path is not None else 0
        observation[self.heading_along_path_index] = min(1.0, max(-1.0, angle_agreement_with_path / math.pi))

        observation[self.cross_track_error_index] = min(1.0, max(-1.0, cross_track_error / (Constants.lane_width * 2)))

        # maxed at 1.0?
        observation[self.yaw_rate_index] = \
            min(1.0, max(-1.0, ego_vehicle.angular_velocity / (.04 * 2 * math.pi)))  # 4

        observation[self.steer_angle_index] = \
            min(1.0, max(-1.0, ego_vehicle.steer_angle / Constants.max_steer_angle))  # 5

        # minned at -1.0?
        observation[self.acceleration_index] = \
            min(1.0, max(-1.0,
                         ego_vehicle.acceleration / (1.1 * Constants.max_acceleration)
                         if ego_vehicle.acceleration >= 0 else
                         ego_vehicle.acceleration / (1.1 * Constants.max_deceleration)))  # 6
        # print('acc: ', ego_vehicle.acceleration, observation[self.acceleration_index])

        # 0'ed
        observation[self.lateral_velocity_index] = \
            min(1.0, max(-1.0, ego_vehicle.lateral_velocity / 1.0))  # 1 m/s cap (7)

        # 1'ed
        observation[self.longitudinal_velocity_index] = \
            min(1.0, max(-1.0, ego_vehicle.longitudinal_velocity / (1.1 * Constants.max_speed)))

        # print(process.distance_to_target, process.target_offset)
        # print(observation[self.distance_to_target_index:self.heading_along_path_index + 1])
        # print(ego_vehicle.lateral_velocity)
        # print(observation[self.distance_to_target_index:self.longitudinal_velocity_index + 1])

        # observation[self.time_left_index] = 2.0 * (process.time / process.time_limit) - 1.0

        yaw = ego_vehicle.angle
        space = process.space

        self.forward_scan_results = \
            self.forward_scanner.scan_closest_of_each_type(
                position,
                yaw - self.forward_scan_offset,
                space,
                lambda rsr: rsr.entity != ego_vehicle)
        self.__extract_scan_data(
            self.forward_scan_results,
            observation,
            self.forward_scan_vehicle_index,
            position,
            velocity)

        self.rear_scan_results = \
            self.rear_scanner.scan_closest_of_each_type(
                position, yaw + self.rear_scan_offset, space,
                lambda rsr: rsr.entity != ego_vehicle)
        self.__extract_scan_data(
            self.rear_scan_results,
            observation,
            self.rear_scan_vehicle_index,
            position,
            velocity)

        # baseline_steer_rate, baseline_jerk = self.get_baseline_action()
        # # observation[self.baseline_acceleration_index] = baseline_jerk / Constants.max_jerk
        # # observation[self.baseline_steer_angle_index] = baseline_steer_rate / Constants.max_steer_rate
        # observation[self.baseline_acceleration_index] = \
        #     inverse_two_sided_exponential(2, baseline_jerk / Constants.max_jerk)
        # observation[self.baseline_steer_angle_index] = \
        #     inverse_two_sided_exponential(2, baseline_steer_rate / Constants.max_steer_rate)
        # # observation[self.baseline_acceleration_index] = baseline_acceleration / Constants.max_acceleration
        # # observation[self.baseline_steer_angle_index] = max_baseline_steer_angle / Constants.max_steer_angle

        if self.num % 1000 == 0:
            print(observation)
        self.num = self.num + 1

        observation = np.multiply(self.scaling, observation)
        # print(ego_position, target_position, delta, observation)
        # print(observation)
        return observation

    def render(self, process: RoadProcess, view: KRoadView) -> None:
        self.draw_scan(process, view, self.forward_scanner, self.forward_scan_results)
        self.draw_scan(process, view, self.rear_scanner, self.rear_scan_results, (96, 0, 0))

    def draw_scan(self, process: RoadProcess, view: KRoadView, ray_scanner, scan_results, color=(64, 64, 64)) -> None:
        if scan_results is None:
            return

        position = process.ego_vehicle.position
        contact_size = min(.1, ray_scanner.beam_radius / 2)
        for (endpoint, ray_contacts) in scan_results:
            ray_vector = (endpoint - position)
            view.draw_segment(color, False, position, endpoint, ray_scanner.beam_radius)

            def draw_contact(color, type_):
                if type_ in ray_contacts:
                    view.draw_circle(
                        color,
                        position + ray_vector * ray_contacts[type_].alpha,
                        contact_size)

            draw_contact((255, 255, 255), EntityType.curb)
            # draw_contact((0, 255, 0), EntityType.lane_marking_white_dashed)
            draw_contact((255, 0, 0), EntityType.vehicle)

    def __extract_scan_data(self, scan_results, observation, index_offset, position, velocity):
        num_rays = len(scan_results)

        for i in range(num_rays):
            (endpoint, ray_contacts) = scan_results[i]

            ray_unit = (endpoint - position).normalized()

            # def get_contact(type_):
            #     if type_ in ray_contacts:
            #         return 2 * ray_contacts[type_].alpha - 1
            #     else:
            #         return 1.0

            def get_contact(type_):
                if type_ in ray_contacts:
                    return 1 - ray_contacts[type_].alpha
                else:
                    return 0.0

            def get_contact_closing_speed(type_):
                if type_ not in ray_contacts:
                    return 0.0
                rcr = ray_contacts[type_]
                relative_velocity = rcr.entity.velocity - velocity
                return max(-1.0, min(1.0, relative_velocity.dot(ray_unit) / Constants.max_speed))

            offset = 0

            # observation[index_offset + num_rays * offset + i] = get_contact(EntityType.vehicle)
            # offset = offset + 1
            #
            # observation[index_offset + num_rays * offset + i] = get_contact_closing_speed(EntityType.vehicle)
            # offset = offset + 1

            observation[index_offset + num_rays * offset + i] = get_contact(EntityType.curb)
            offset = offset + 1

            # observation[index_offset + num_rays * offset + i] = get_contact_closing_speed(EntityType.curb)
            # offset = offset + 1

            # try closing time as well?
            # closing time to vehicle vs along ray? maybe essentially the same?

            # observation[index_offset + num_rays * offset + i] = get_contact(EntityType.lane_marking_white_dashed)
            # offset = offset + 1
