from typing import (
    TYPE_CHECKING,
    Tuple,
)

import pymunk
from pymunk import (
    Arbiter,
    Vec2d,
)

from k_road.constants import Constants
from k_road.entity.entity_category import EntityCategory
from k_road.entity.vehicle.vehicle import Vehicle
from k_road.util import *

if TYPE_CHECKING:
    from k_road.k_road_view import KRoadView
    from k_road.k_road_process import KRoadProcess


class SingleBodyVehicle(Vehicle):
    """
    A single-body vehicle
    """

    def __init__(self,
                 parent: 'KRoadProcess',
                 position: Vec2d,
                 body_velocity: Vec2d,
                 yaw: float,
                 color: Tuple[int, int, int],
                 length: float = 4.9,  # a typical car length (Toyota Camry)
                 width: float = 1.83,  # a typical car width (Toyota Camry)
                 steer_angle: float = 0.0,
                 max_speed=Constants.max_speed,
                 max_steer_angle: float = reset_angle((60 / 360.0) * (pi * 2)),
                 max_acceleration: float = Constants.max_acceleration,
                 max_deceleration: float = Constants.max_deceleration,
                 angular_velocity: float = 0  # rotational velocity of the vehicle
                 ):
        self.length: float = length
        self.width: float = width

        body = pymunk.Body(0, 0, pymunk.Body.KINEMATIC)
        half_length: float = self.length / 2
        half_width: float = self.width / 2

        # although shape isn't referenced, assigning it here prevents GC of the shape before it is added to the space
        shape = pymunk.Poly(body,
                            [(half_length, half_width),
                             (-half_length, half_width),
                             (-half_length, -half_width),
                             (half_length, -half_width)],
                            None,
                            radius=.1)
        body.position = position
        body.angle = yaw

        super().__init__(parent, body, color)

        self.__body_velocity: Vec2d = Vec2d(0, 0)
        self.max_speed: float = max_speed
        # self.position: Vec2d = position
        self.acceleration: float = 0.0
        self.steer_angle: float = steer_angle  # relative to the body frame
        self.max_steer_angle: float = max_steer_angle
        self.set_velocity_and_yaw(body_velocity, yaw)
        self.max_acceleration: float = max_acceleration
        self.max_deceleration: float = max_deceleration
        self.angular_velocity = angular_velocity

    def set_velocity_and_yaw(self, body_velocity: Vec2d, yaw: float) -> None:
        """
        Sets velocity and yaw (angle).
        It's more efficient to do both at once (since world velocity depends on both), so this setter sets both at once.
        """

        self.__body_velocity = body_velocity
        Vehicle.angle.fset(self, yaw)
        self.velocity = body_velocity.rotated(yaw)
        # self.velocity = Vec2d(
        #     body_velocity[0] * math.cos(self.angle) - body_velocity[1] * math.sin(self.angle),
        #     body_velocity[0] * math.sin(self.angle) + body_velocity[1] * math.cos(self.angle))
        # if math.fabs(body_velocity[1]) > 1e-3 or math.fabs(yaw) > 1e-3:
        #     print('bv: ', body_velocity, ' yaw ', yaw, ' a ', self.angle)

    @property
    def positions(self) -> (Vec2d, Vec2d, Vec2d):
        return self.front_position, self.position, self.rear_position

    @property
    def front_position(self) -> Vec2d:
        return self.position + Vec2d(self.length / 2, 0).rotated(self.angle)

    @property
    def rear_position(self) -> Vec2d:
        return self.position + Vec2d(-self.length / 2, 0).rotated(self.angle)

    @property
    def body_velocity(self) -> Vec2d:
        return self.__body_velocity

    @body_velocity.setter
    def body_velocity(self, body_velocity: Vec2d) -> None:
        self.set_velocity_and_yaw(body_velocity, self.angle)

    @property
    def longitudinal_velocity(self) -> float:
        return self.body_velocity[0]

    @property
    def lateral_velocity(self) -> float:
        return self.body_velocity[1]

    @Vehicle.angle.setter
    def angle(self, yaw: float) -> None:
        self.set_velocity_and_yaw(self.body_velocity, yaw)

    def render(self, view: 'KRoadView') -> None:
        # TODO: draw steer angle / wheels
        view.draw_entity(self, self.color, False)
        # view.draw_segment((0, 128, 255), False, self.scan_start, self.scan_endpoint, self.collision_scan_radius)
        # if self.contact is not None:
        #     view.draw_circle((255, 128, 0), self.contact.point, .5)

    def handle_collision(self, arbiter: Arbiter, other_entity: 'Entity') -> bool:
        other_category = other_entity.category
        if other_category == EntityCategory.dynamic or other_category == EntityCategory.off_road:
            print('ego collision: ',
                  other_entity.type_,
                  other_entity.category,
                  other_entity.id,
                  self.id,
                  self.parent.time,
                  # other_entity.position, self.position,
                  # other_entity.velocity, self.velocity,
                  other_entity.velocity.length, self.velocity.length,
                  other_entity.velocity.angle, self.velocity.angle)
            self.collided = True
        return True
