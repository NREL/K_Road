from typing import (
    List,
    Union,
)

import pymunk

from k_road.entity.entity import Entity
from k_road.entity.entity_type import EntityType


class DynamicEntity(Entity):

    def __init__(self,
                 type_: EntityType,
                 parent: 'KRoadProcess',
                 bodies: Union[pymunk.Body, List[pymunk.Body]]):
        super().__init__(type_, parent, bodies)
        if self.parent:
            self.parent.set_dynamic(self)
