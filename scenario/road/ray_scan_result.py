import pymunk
from pymunk import Vec2d

from k_road.entity.entity import Entity
from k_road.entity.entity_type import EntityType


class RayScanResult:
    __slots__ = 'alpha', 'type_', 'entity_id', 'point', 'normal', 'shape', 'entity'

    def __init__(self, alpha: float, type_: EntityType, entity_id: int, point: Vec2d, normal: Vec2d,
                 shape: pymunk.Shape, entity: Entity):
        self.alpha: float = alpha
        self.type_: EntityType = type_
        self.entity_id: int = entity_id
        self.point: Vec2d = point
        self.normal: Vec2d = normal
        self.shape: pymunk.Shape = shape
        self.entity: Entity = entity

    def __repr__(self):
        return (self.alpha, self.type_, self.entity_id, self.point, self.normal,
                self.shape, self.entity).__repr__()
