import math
import random
from typing import Any

from pymunk import Vec2d

from factored_gym import Process
from k_road.entity.target import Target
from k_road.entity.vehicle.vehicle_dbm3 import VehicleDBM3
from k_road.k_road_view import KRoadView
from k_road.single_vehicle_process import SingleVehicleProcess


class TargetProcess(SingleVehicleProcess):
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self,
                 min_starting_distance: float = 50,
                 max_starting_distance: float = 100,
                 time_step_length: float = .05,
                 **kwargs):
        super().__init__(time_step_length, **kwargs)

        # ---- simulation properties ----
        self.max_starting_distance: float = max_starting_distance
        self.min_starting_distance: float = min_starting_distance
        self.max_distance_from_target: float = max_starting_distance + 50

        # ---- state vars ----
        self.target = None

        self.ego_to_target = None
        self.distance_to_target: float = 0

    def make_view(self, mode) -> KRoadView:
        scale = 4.0
        padding = .01
        size = 2 * self.max_distance_from_target * (1.0 + padding)
        view_size = math.ceil(scale * size)

        return KRoadView(
            (view_size, view_size),
            Vec2d(0, 0),
            scale,
            self.time_dilation)

    def reset(self, process: Process) -> None:
        super().reset(self)

        for entity in list(self.entities):
            entity.discard()

        self.target = Target(self, Vec2d(0, 0))

        distance_to_target = random.uniform(self.min_starting_distance, self.max_starting_distance)
        heading_to_target = random.uniform(-math.pi, math.pi)
        starting_position = Vec2d(0, distance_to_target).rotated(heading_to_target)

        self.ego_vehicle = VehicleDBM3(
            self,
            (255, 64, 64),
            starting_position,
            Vec2d(0, 0),
            yaw=random.uniform(-math.pi, math.pi),
        )

        self._update_delta_vector()

    def step(self, action) -> Any:
        self.ego_vehicle.apply_normalized_control_action(self.time_step_length, action)
        super().step(action)
        self._update_delta_vector()
        return {}

    def _update_delta_vector(self):
        ego_position = self.ego_vehicle.position
        target_position = self.target.position
        self.ego_to_target = target_position - ego_position
        self.distance_to_target = self.ego_to_target.get_length() - self.target.radius
