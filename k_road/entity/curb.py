from typing import TYPE_CHECKING

from pymunk import Vec2d

from k_road.entity.entity_type import EntityType
from k_road.entity.linear_entity import LinearEntity

if TYPE_CHECKING:
    from k_road.k_road_process import KRoadProcess


class Curb(LinearEntity):
    """
    """

    def __init__(self, parent: 'KRoadProcess', start: Vec2d, end: Vec2d, radius: float = .1, **kwargs):
        super().__init__(EntityType.curb, parent, start, end, radius=radius, color=(128, 128, 128), **kwargs)
