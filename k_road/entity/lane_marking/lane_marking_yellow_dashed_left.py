from typing import TYPE_CHECKING

from pymunk import Vec2d

from k_road.entity.entity_type import EntityType
from k_road.entity.lane_marking.lane_marking import LaneMarking

if TYPE_CHECKING:
    from k_road.k_road_process import KRoadProcess


class LaneMarkingYellowDashedLeft(LaneMarking):
    """
    A double yellow lane line with a dashed left line.
    """

    def __init__(self,
                 parent: 'KRoadProcess',
                 start: Vec2d,
                 end: Vec2d,
                 **kwargs):
        super().__init__(
            EntityType.lane_marking_yellow_dashed_left,
            parent,
            start,
            end,
            LaneMarking.yellow_color,
            LaneMarking.double_width,
            **kwargs)

    def render(self, view: 'KRoadView') -> None:
        self.draw_offset_dashed_line(view, -LaneMarking.double_offset)
        self.draw_offset_line(view, LaneMarking.double_offset)
