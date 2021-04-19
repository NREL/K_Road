from typing import (
    List,
    TYPE_CHECKING,
    Tuple,
    Union,
)

import pymunk

from k_road.entity.dynamic_entity import DynamicEntity
from k_road.entity.entity_type import EntityType

if TYPE_CHECKING:
    from k_road.k_road_process import KRoadProcess


class Vehicle(DynamicEntity):
    """
    A vehicle
    """

    def __init__(self,
                 parent: 'KRoadProcess',
                 bodies: Union[pymunk.Body, List[pymunk.Body]],
                 color: Tuple[int, int, int],
                 ):
        super().__init__(EntityType.vehicle, parent, bodies)
        self.color = color

    def is_in_valid_state(self) -> bool:
        return True
