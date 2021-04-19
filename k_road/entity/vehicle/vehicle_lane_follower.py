from typing import (
    Optional,
    TYPE_CHECKING,
    Tuple,
)

import pymunk
from pymunk import Vec2d

from k_road.constants import Constants
from k_road.entity.entity import Entity
from k_road.entity.entity_category import EntityCategory
from k_road.entity.vehicle.single_body_vehicle import SingleBodyVehicle
from k_road.model.intelligent_driver_model import IntelligentDriverModel
from k_road.scan.waypoint_path_follower import WaypointPathFollower

if TYPE_CHECKING:
    from k_road.k_road_process import KRoadProcess


class VehicleLaneFollower(SingleBodyVehicle):
    """
    A vehicle that follows a lane path.
    """

    def __init__(self,
                 parent: 'KRoadProcess',
                 color: Tuple[int, int, int],
                 position: Vec2d,
                 body_velocity: Vec2d,
                 desired_speed: float,
                 yaw: float,
                 longitudinal_model: IntelligentDriverModel = IntelligentDriverModel(),
                 max_speed: float = Constants.max_speed,
                 **kwargs
                 ):
        # TODO: rearrange args to be consistent with superclass
        super().__init__(parent, position, body_velocity, yaw, color, **kwargs)

        self.desired_speed: float = desired_speed
        self.max_speed = max_speed
        self.longitudinal_model: IntelligentDriverModel = longitudinal_model
        self.idm = IntelligentDriverModel()

        self.lane_center: Vec2d = position
        self.left_contact: Optional[Tuple[float, Vec2d, Vec2d, Entity]] = None
        self.right_contact: Optional[Tuple[float, Vec2d, Vec2d, Entity]] = None

        self.path_scan_radius: float = max(Constants.lane_width, self.length * 2)
        self.collision_scan_distance: float = 100
        self.collision_scan_radius: float = Constants.lane_width * .5

        self.scan_start = self.position
        self.scan_endpoint = self.position
        self.contact = None

        self.waypoint_path_follower: WaypointPathFollower = \
            WaypointPathFollower(self.parent.space, self.position, Constants.lane_width * .7, start_mid_path=True)

    def render(self, view: 'KRoadView') -> None:
        # view.draw_segment((0, 128, 255), False, self.scan_start, self.scan_endpoint, self.collision_scan_radius)
        # if self.contact is not None:
        #     view.draw_circle((255, 128, 0), self.contact.point, .5)

        super().render(view)

    def step(self) -> None:
        """
        scans for the nearest path in front of the vehicle and follows it
        """
        self.contact = None

        position: Vec2d = self.position
        space: pymunk.Space = self.parent.space
        yaw: float = self.angle

        time_step_length: float = self.parent.time_step_length
        dx: float = time_step_length * self.desired_speed

        if (self.position - self.waypoint_path_follower.position).length > .1:
            self.position = self.waypoint_path_follower.position

        follower_state, next_position = self.waypoint_path_follower.get_next_waypoint(dx)
        if follower_state != WaypointPathFollower.State.searching_for_path_start:
            delta_position = next_position - position
            velocity = delta_position * (1.0 / time_step_length)
            next_yaw = delta_position.angle
            self.set_velocity_and_yaw(Vec2d(velocity.length, 0), next_yaw)

        # scan in front of the vehicle to get any vehicles in the way
        yaw = self.angle
        position = self.position
        velocity = self.velocity
        direction = Vec2d(1, 0).rotated(yaw)
        speed = velocity.length
        shape_filter = pymunk.ShapeFilter(categories=EntityCategory.dynamic_sensor)
        self.scan_start = position - Vec2d(self.length / 2, 0).rotated(yaw)
        self.scan_endpoint = position + direction * self.collision_scan_distance
        sqis = space.segment_query(self.scan_start, self.scan_endpoint, self.collision_scan_radius, shape_filter)

        closest_obstacle = None
        closest_obstacle_distance = self.collision_scan_distance + self.collision_scan_radius - self.length / 2
        for sqi in sqis:
            if sqi.shape is None or sqi.shape.body.entity == self:
                continue

            shape = sqi.shape
            entity = shape.body.entity
            # type_ = entity.type_
            category = entity.category
            delta = sqi.point - position
            distance = delta.length
            if category == EntityCategory.off_road or category == EntityCategory.dynamic:
                if closest_obstacle_distance > distance:
                    closest_obstacle = sqi
                    closest_obstacle_distance = distance

        closest_obstacle_distance = max(0.0, closest_obstacle_distance - self.length / 2)
        acceleration = 0
        if closest_obstacle is None:
            acceleration = self.longitudinal_model.get_free_acceleration(self.desired_speed, speed)
            # print('idm free ', self.id, acceleration, self.desired_speed, speed)
        else:
            self.contact = closest_obstacle
            # print('contact ', self.id, closest_obstacle.shape.body.entity.id, closest_obstacle_distance)
            closest_entity = closest_obstacle.shape.body.entity
            acceleration = self.longitudinal_model.get_acceleration(
                self.desired_speed,
                speed,
                closest_entity.speed,
                closest_obstacle_distance)

            # if closest_obstacle_distance < 5:
            #     print('idm interaction ',
            #           self.id, acceleration,
            #           self.desired_speed,
            #           speed,
            #           closest_entity.speed,
            #           closest_obstacle_distance)

        speed_delta = acceleration * time_step_length
        new_speed = max(0, speed + speed_delta)
        new_velocity = direction * new_speed
        # print('set idm velocity ', self.id, acceleration, velocity, time_step_length,
        #       velocity * (1.0 + acceleration * time_step_length))

        self.velocity = new_velocity
