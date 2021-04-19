from typing import (
    TYPE_CHECKING,
)

from pymunk import Vec2d

from k_road.entity.entity_type import EntityType
from k_road.entity.linear_entity import LinearEntity

if TYPE_CHECKING:
    from k_road.k_road_view import KRoadView
    from k_road.k_road_process import KRoadProcess


class Path(LinearEntity):
    """
    A path that vehicles can follow, like down a road.
    The path is one way and heads from start to end.
    """

    def __init__(self, parent: 'KRoadProcess', start: Vec2d, end: Vec2d, target_speed: float = 11.1,
                 color=(255, 255, 128), **kwargs):
        super().__init__(EntityType.path, parent, start, end, radius=0, **kwargs)
        self.target_speed: float = target_speed

    def render(self, view: 'KRoadView') -> None:
        view.draw_line(self.color, self.start, self.end)
