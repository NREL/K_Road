from typing import (
    List,
    Optional,
    TYPE_CHECKING,
)

import pymunk
from pymunk import (
    Segment,
    Vec2d,
)

from k_road.entity.entity import Entity
from k_road.entity.entity_type import EntityType

if TYPE_CHECKING:
    from k_road.k_road_view import KRoadView
    from k_road.k_road_process import KRoadProcess


class LinearEntity(Entity):
    """
    Common class for linear static entities like curbs, lane markings, and paths.
    """

    def __init__(self,
                 type_: EntityType,
                 parent: 'KRoadProcess',
                 start: Vec2d,
                 end: Vec2d,
                 radius: float = 1.0,
                 color: (int, int, int) = (128, 128, 128),
                 # group_id: Union[int, str, None] = None,
                 sequence_id: int = 0,
                 sequence_length: int = 0,
                 sequence: Optional[List['LinearEntity']] = None,
                 ):
        self.color: (int, int, int) = color
        # self.group_id: Union[int, str, None] = group_id
        self.sequence: Optional[List[LinearEntity]] = sequence
        self.sequence_id = sequence_id
        self.sequence_length: int = sequence_length
        self.start: Vec2d = start
        self.end: Vec2d = end

        body = pymunk.Body(mass=0, moment=0, body_type=pymunk.Body.STATIC)
        self.delta: Vec2d = end - start

        # although shape isn't referenced, assigning it here prevents GC of the shape before it is added to the space
        shape: Segment = pymunk.Segment(body, Vec2d(0, 0), Vec2d(self.delta.length, 0), radius)

        body.position = start
        body.angle = self.delta.angle

        super().__init__(type_, parent, body)

    def render(self, view: 'KRoadView') -> None:
        view.draw_entity(self, self.color, True)

    @property
    def length(self) -> float:
        return self.delta.length

    @property
    def direction(self) -> Vec2d:
        return self.delta.normalized()

    @property
    def perpendicular(self) -> Vec2d:
        return self.direction.perpendicular()
