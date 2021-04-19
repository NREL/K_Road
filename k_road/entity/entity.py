from abc import ABC
from typing import (
    List,
    TYPE_CHECKING,
    Union,
)

import pymunk
from pymunk import Vec2d

from k_road.entity.entity_category import EntityCategory
from k_road.entity.entity_type import EntityType

if TYPE_CHECKING:
    from k_road.k_road_view import KRoadView
    from k_road.k_road_process import KRoadProcess


class Entity(ABC):

    # __slots__ = 'id', 'type_', 'parent', 'bodies'

    def __init__(self,
                 type_: EntityType,
                 parent: 'KRoadProcess',
                 bodies: Union[pymunk.Body, List[pymunk.Body]],
                 ):
        self.type_: EntityType = type_
        self.id: int = 0  # set when added to a KRoadProcess
        self.parent: 'KRoadProcess' = parent
        self.live: bool = True

        if isinstance(bodies, pymunk.Body):
            bodies = [bodies]

        self.bodies: [pymunk.Body] = bodies
        if self.parent:
            self.parent.add_entity(self)

    @property
    def body(self) -> pymunk.Body:
        """Convenience function to get zeroth body."""
        return self.bodies[0]

    @property
    def category(self) -> EntityCategory:
        """ Convenience function to get entity category. """
        return self.type_.category

    @property
    def shapes(self) -> [pymunk.shapes]:
        """ Convenience function to get all shapes of this entity. """
        return [shape for body in self.bodies for shape in body.shapes]

    def render(self, view: 'KRoadView') -> None:
        view.draw_entity(self)

        # for collision in self.collisions:
        #     for point in collision.contact_point_set:
        #         a = point.point_a
        #         b = point.point_b
        #         view.draw_circle((255, 0, 0), a, .1)
        #         view.draw_circle((255, 0, 0), b, .1)

    def discard(self) -> None:
        """Removes this entity from the simulation"""
        if self.live:
            self.live = False
            self.parent.discard_entity(self)

    def step(self) -> None:
        pass

    def handle_collision(self, arbiter: pymunk.Arbiter, other_entity: 'Entity') -> bool:
        return other_entity != self

    @property
    def has_single_body(self) -> bool:
        return len(self.bodies) == 1

    @property
    def position(self) -> Vec2d:
        return self.body.position

    @position.setter
    def position(self, position: Vec2d) -> None:
        if self.has_single_body:
            self.body.position = position
        else:
            delta = position - self.position
            for body in self.bodies:
                body.position = body.position + delta

    @property
    def velocity(self) -> Vec2d:
        return self.body.velocity

    @velocity.setter
    def velocity(self, velocity: Vec2d) -> None:
        for body in self.bodies:
            body.velocity = velocity

    @property
    def speed(self) -> float:
        return self.velocity.get_length()

    @property
    def angle(self) -> float:
        return self.body.angle

    @angle.setter
    def angle(self, angle: float) -> None:
        if self.has_single_body:
            self.body.angle = angle
        else:
            delta = angle - self.angle
            for body in self.bodies:
                body.angle = body.angle + delta

    @property
    def angular_velocity(self) -> float:
        return self.body.angular_velocity

    @angular_velocity.setter
    def angular_velocity(self, angular_velocity: float) -> None:
        for body in self.bodies:
            body.angular_velocity = angular_velocity

    # @property
    # def mass(self) -> float:
    #     return sum([body.mass for body in self.bodies])
    #
    # @mass.setter
    # def mass(self, mass: float) -> None:
    #     if self.has_single_body:
    #         self.body.mass = mass
    #     else:
    #         scale = mass / self.mass
    #         for body in self.bodies:
    #             body.mass = scale * body.mass
    #
    # @property
    # def moment(self) -> float:
    #     """
    #     moment of inertia
    #     """
    #     return self.body.moment
    #
    # @moment.setter
    # def moment(self, moment: float) -> None:
    #     if self.has_single_body:
    #         self.body.moment = moment
    #     else:
    #         scale = moment / self.moment
    #         for body in self.bodies:
    #             body.moment = scale * body.moment
