from typing import TYPE_CHECKING

import pymunk
from pymunk import Vec2d

from k_road.entity.entity import Entity
from k_road.entity.entity_type import EntityType

if TYPE_CHECKING:
    from k_road.k_road_view import KRoadView


class Target(Entity):

    def __init__(self, parent, position, radius=1):
        self.radius = radius

        body = pymunk.Body(0, 0, pymunk.Body.STATIC)
        shape = pymunk.Circle(body, radius)
        body.position = position
        shape.sensor = True
        # body.velocity = velocity

        super().__init__(EntityType.target, parent, body)

    def render(self, view: 'KRoadView'):
        view.draw_circle_on_entity(self, (64, 225, 64), False, Vec2d(0, 0), self.radius)
        if self.radius < 5:
            view.draw_circle_on_entity(self, (64, 225, 64), False, Vec2d(0, 0), 5)
