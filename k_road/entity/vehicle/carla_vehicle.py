from typing import (
    List,
    Tuple,
    Union,
)

import pymunk

from k_road.entity.vehicle.carla_manager import CarlaManager
from k_road.entity.vehicle.vehicle import Vehicle


class CarlaVehicle(Vehicle):

    def __init__(self,
                 manager: CarlaManager,
                 id: int,
                 parent: 'KRoadProcess',
                 bodies: Union[pymunk.Body, List[pymunk.Body]],
                 color: Tuple[int, int, int],
                 ):
        self.manager: CarlaManager = manager
        self.id: int = id
        super().__init__(parent, bodies, color)
