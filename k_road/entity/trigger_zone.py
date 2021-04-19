from typing import TYPE_CHECKING

from pymunk import (
    Arbiter,
    Vec2d,
)

from k_road.entity.entity_category import EntityCategory
from k_road.entity.entity_type import EntityType
from k_road.entity.linear_entity import LinearEntity

if TYPE_CHECKING:
    pass


class TriggerZone(LinearEntity):

    def __init__(self, parent: 'KRoadProcess', start: Vec2d, end: Vec2d, on_collision, radius: float = .1, **kwargs):
        super().__init__(EntityType.trigger_zone, parent, start, end, radius, (255, 32, 32), **kwargs)
        self.on_collision = on_collision
        # for shape in self.get_shapes():
        #     shape.sensor = True

    def handle_collision(self, arbiter: Arbiter, other_entity: 'Entity') -> bool:
        if other_entity.category == EntityCategory.dynamic:
            if self.on_collision(self, arbiter, other_entity):
                other_entity.discard()
                return False

        return super().handle_collision(arbiter, other_entity)
