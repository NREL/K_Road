from typing import TYPE_CHECKING

from pymunk import Vec2d

from k_road.entity.entity_type import EntityType
from k_road.entity.lane_marking.lane_marking import LaneMarking

if TYPE_CHECKING:
    from k_road.k_road_process import KRoadProcess


class LaneMarkingWhite(LaneMarking):
    """
    A solid white lane line.
    """

    def __init__(self,
                 parent: 'KRoadProcess',
                 start: Vec2d,
                 end: Vec2d,
                 **kwargs):
        super().__init__(EntityType.lane_marking_white, parent, start, end, LaneMarking.white_color, **kwargs)
