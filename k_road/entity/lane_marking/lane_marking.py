from typing import TYPE_CHECKING

from pymunk import Vec2d

from k_road.entity.entity_category import EntityCategory
from k_road.entity.entity_type import EntityType
from k_road.entity.linear_entity import LinearEntity
from k_road.k_road_process import KRoadView

if TYPE_CHECKING:
    from k_road.k_road_process import KRoadProcess

"""
Lane/Lanelet plan:

+ Each lane marking knows its left and right lanes (if they exist)
+ Each lane knows its direction (or directions, or none), and other attributes (bus only, bike lane, parking lane,
speed limit, etc)
+ Areas indicate areas that aren't lanes. Areas have attributes as well.
+ Are lanes also areas?
+ Lanes also refer to traffic signals, stop signs, etc
+ Lanes know their borders (lane markings)
+ Lane markings know their type (solid, dashed left, dashed right, dashed, yellow, white, red, blue, etc)

+ Entity with multiple shapes? Multiple bodies?

"""


class LaneMarking(LinearEntity):
    """
    a typical lane marking is 4 to 6 inches wide. We use .1m which is approximately 4 inches.
    https://mutcd.fhwa.dot.gov/pdfs/millennium/06.14.01/3ndi.pdf page 3A-3
    
    The colors of longitudinal pavement markings shall conform to the following
    basic concepts:
    A. Yellow lines delineate:
        1. The separation of traffic traveling in opposite directions.
        2. The left edge of the roadways of divided and one-way highways and ramps.
        3. The separation of two-way left turn lanes and reversible lanes from other lanes.
    B. White lines delineate:
        1. The separation of traffic flows in the same direction.
        2. The right edge of the roadway.
    C. Red markings delineate roadways that shall not be entered or used.
    D. Blue markings delineate parking spaces for persons with disabilities
    """

    white_color = (204, 204, 204)
    yellow_color = (204, 204, 0)
    line_width = .1
    double_width = .4
    double_offset = .1

    def __init__(
            self,
            type_: EntityType,
            parent: 'KRoadProcess',
            start: Vec2d,
            end: Vec2d,
            color: (int, int, int),
            radius: float = line_width,
            **kwargs):
        super().__init__(type_, parent, start, end, radius=radius, color=color, **kwargs)
        assert type_.category == EntityCategory.lane_marking, 'Only lane marking entity types are allowed here.'

        # self.left_lane: Optional['Lane'] = None
        # self.right_lane: Optional['Lane'] = None

    def render(self, view: 'KRoadView') -> None:
        self.draw_offset_line(view, 0.0)

    def render_dashed(self, view: 'KRoadView') -> None:
        self.draw_offset_dashed_line(view, 0.0)

    def render_double(self, view: 'KRoadView') -> None:
        self.draw_offset_line(view, LaneMarking.double_offset)
        self.draw_offset_line(view, -LaneMarking.double_offset)

    def get_offset_lines(
            self,
            offset: float
    ) -> [(Vec2d, Vec2d)]:

        def get_segment_offsets(body, segment):
            offset_vector = self.perpendicular * offset

            def get_offset_world_vector(vector):
                return body.local_to_world(vector + offset_vector)

            return get_offset_world_vector(segment.a), get_offset_world_vector(segment.b)

        return [get_segment_offsets(body, segment) for body in self.bodies for segment in body.shapes]

    def draw_offset_line(
            self,
            view: 'KRoadView',
            offset: float,
            radius: float = line_width
    ) -> None:
        for a, b in self.get_offset_lines(offset):
            view.draw_segment(self.color, True, a, b, radius)

    def draw_offset_dashed_line(
            self,
            view: 'KRoadView',
            offset: float,
            radius: float = line_width
    ) -> None:
        for a, b in self.get_offset_lines(offset):
            view.draw_segment_dashed(self.color, True, a, b, radius, 19, .526)
