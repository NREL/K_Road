import math
import random
from typing import (
    Any,
    Optional,
    Union,
)

from pymunk import (
    Arbiter,
    Vec2d,
)

from k_road.builder.road_builder import RoadBuilder
from k_road.entity.entity import Entity
from k_road.entity.entity_type import EntityType
from k_road.entity.trigger_zone import TriggerZone
from k_road.entity.vehicle.vehicle_dbm3 import VehicleDBM3
from k_road.entity.vehicle.vehicle_lane_follower import VehicleLaneFollower
from k_road.single_vehicle_process import SingleVehicleProcess
from k_road.travel_direction import TravelDirection

"""
TODO:
    + Configurable road start & end points
    + Investigate lanelet based adaptations
    + Add other dynamic entities:
        + Pedestrians
        + Animals
        + Bicycles
        + Motorcycles
        + Different vehicle classes
    + Oncoming Traffic
    + Parking
    + Stalling
"""


# np.set_printoptions(linewidth=200, precision=3, threshold=np.nan)


class RoadProcess(SingleVehicleProcess):
    metadata = {
        'render.modes': ['human']
    }
    default_lane_layout: [Union[EntityType, TravelDirection]] = [
        EntityType.curb,
        EntityType.lane_marking_yellow,
        EntityType.lane,
        EntityType.lane_marking_white_dashed,
        EntityType.lane,
        EntityType.lane_marking_white_dashed,
        EntityType.lane,
        EntityType.lane_marking_yellow_double,
        EntityType.lane,
        EntityType.lane_marking_white_dashed,
        EntityType.lane,
        EntityType.lane_marking_white_dashed,
        EntityType.lane,
        EntityType.lane_marking_yellow,
        EntityType.curb
    ]

    # default_lane_layout: [Union[EntityType, TravelDirection]] = [
    #     EntityType.curb,
    #     EntityType.lane_marking_yellow,
    #     EntityType.lane,
    #     EntityType.lane_marking_yellow,
    #     EntityType.curb
    #     ]

    # default_lane_layout: [Union[EntityType, TravelDirection]] = [
    #     EntityType.curb,
    #     EntityType.lane_marking_yellow,
    #     EntityType.lane,
    #     20.0,
    #     EntityType.lane_marking_yellow,
    #     EntityType.curb
    #     ]

    # default_road_shape = [Vec2d(0, 0), Vec2d(800, 0)]

    #  default_road_shape = [Vec2d(0, 0), Vec2d(550, 500), Vec2d(1000, 1000)]
    #  default_road_shape = [Vec2d(0, 0), Vec2d(100, 0), Vec2d(140, 40), Vec2d(150, 50), Vec2d(150, 100)]
    # default_road_shape = [Vec2d(0, 0), Vec2d(200, 0), Vec2d(200, 50)]
    default_road_shape = [Vec2d(1000 * (i / 100.0), 20.0 * math.sin(2 * (i / 100.0) * 2 * math.pi)) for i in range(
        100)]

    # default_road_shape = [Vec2d(250 * (i / 100.0), 20.0 * math.sin(1 * (i / 100.0) * 2 * math.pi)) for i in range(
    # 100)]

    def __init__(self,
                 start: Vec2d = Vec2d(0, 0),
                 end: Vec2d = Vec2d(800, 200),
                 lane_layout: Optional[Union[EntityType, TravelDirection]] = None,
                 road_shape: Optional[Vec2d] = None,
                 ego_starting_distance: float = 200.0,
                 speed_mean: float = 22.3,  # 50 mph
                 speed_standard_deviation: float = 4.47,  # 10mph
                 traffic_density: float = .04,  # .04,
                 target_inset: float = 50,  # = 200.0,
                 **kwargs
                 ):

        if lane_layout is None:
            lane_layout = RoadProcess.default_lane_layout
        if road_shape is None:
            road_shape = RoadProcess.default_road_shape

        super().__init__(
            view_follows_ego=False,
            **kwargs)

        # ---- simulation properties ----

        # self.road = RoadBuilder(self, None, [start, end], None, lane_layout)
        self.road = self.build_road()

        self.ego_starting_distance: float = ego_starting_distance
        self.speed_mean: float = speed_mean
        self.speed_standard_deviation: float = speed_standard_deviation
        self.traffic_density: float = traffic_density
        self.target_offset: float = self.road.length - target_inset
        self.target_num_vehicles: int = \
            int(self.traffic_density * self.road.num_lanes * self.road.length)
        self.num_vehicles: int = 0

        self.distance_to_target: float = 0.0
        self.ego_in_end_zone: bool = False

    def build_road(self):
        self.road = RoadBuilder(
            self,
            None,
            self.road_shape,
            None,
            self.lane_layout)
        return self.road

    def set_starting_distance(self, starting_distance: float):
        self.ego_starting_distance = starting_distance
        return self.ego_starting_distance

    def reset(self, process: 'RoadProcess') -> None:
        for entity in list(self.dynamic_entities):
            entity.discard()

        super().reset(self)
        self.step_physics()

        # where the ego vehicle ends its run
        road = self.road
        TriggerZone(self,
                    road.road_to_global(Vec2d(self.target_offset, 0))[0],
                    road.road_to_global(Vec2d(self.target_offset, road.width))[0],
                    lambda end_zone, arbiter, entity: self.handle_ego_zone_collision(end_zone, arbiter, entity))

        # where the bot vehicles end their run
        TriggerZone(self,
                    road.road_to_global(Vec2d(road.length, 0))[0],
                    road.road_to_global(Vec2d(road.length, road.width))[0],
                    lambda end_zone, arbiter, entity: self.handle_end_zone_collision(end_zone, arbiter, entity))

        self.ego_in_end_zone = False
        self.place_ego_vehicle()

        # self.target_num_vehicles = int(self.traffic_density * self.road.num_lanes * self.road.length)

        self.num_vehicles = 0
        for i in range(self.target_num_vehicles):
            self.try_spawn_vehicle(random.uniform(0, self.road.length))

        self._update_state_vars()

    def step(self, action) -> Any:
        # print(action)

        self.ego_vehicle.apply_control(self.time_step_length, action[0], action[1])

        if self.ego_in_end_zone:
            self.reset(self)

        super().step(action)

        self._update_state_vars()
        if self.num_vehicles < self.target_num_vehicles:
            self.try_spawn_vehicle()
        return {}

    def try_spawn_vehicle(self, offset: float = 0) -> bool:
        # print('try_spawn_vehicle')
        velocity = Vec2d(random.gauss(self.speed_mean, self.speed_standard_deviation), 0)
        position, angle = self.road.road_to_global(
            Vec2d(offset,
                  (self.road.lane_center_offsets[random.randrange(0, self.road.num_lanes)])))

        vehicle = VehicleLaneFollower(
            self,
            (64, 64, 200),
            position,
            velocity,
            velocity.length,
            angle)

        if not self.can_place_entity(vehicle, 10):
            # print('remove ', position, angle)
            vehicle.discard()
            # print('try_spawn_vehicle remove ', vehicle.id)
            return False

        # print('try_spawn_vehicle add ', vehicle.id, vehicle.position, vehicle.velocity)
        # vehicle.handle_collision =
        self.num_vehicles = self.num_vehicles + 1
        return True

    def _update_state_vars(self) -> None:
        self.distance_to_target = self.target_offset - self.road.global_to_road(self.ego_vehicle.position).x

    def handle_ego_zone_collision(self, zone: TriggerZone, arbiter: Arbiter, entity: Entity):
        if entity == self.ego_vehicle:
            self.ego_in_end_zone = True
        return False

    def handle_end_zone_collision(self, zone: TriggerZone, arbiter: Arbiter, entity: Entity):
        if entity == self.ego_vehicle:
            return False
        self.num_vehicles = self.num_vehicles - 1
        return True

    def place_ego_vehicle(self):
        if self.ego_vehicle is not None:
            self.ego_vehicle.discard()
            self.ego_vehicle = None

        while not self.try_reset_ego_vehicle():
            self.step_physics()

    def try_reset_ego_vehicle(self) -> bool:
        position, angle = self.road.road_to_global(
            Vec2d(self.target_offset - self.ego_starting_distance,
                  self.road.lane_center_offsets[random.randrange(0, self.road.num_lanes)]))

        # vehicle = VehicleDDBM(
        #     self,
        #     (200, 64, 64),
        #     position,
        #     # Vec2d(self.road_buffer_space, (.5 + 2) * Constants.lane_width),
        #     Vec2d(random.gauss(self.speed_mean, self.speed_standard_deviation), 0),
        #     angle)

        vehicle = VehicleDBM3(
            self,
            (200, 64, 64),
            position,
            # Vec2d(self.road_buffer_space, (.5 + 2) * Constants.lane_width),
            Vec2d(0, 0),
            angle)

        if not self.can_place_entity(vehicle, 10):
            vehicle.discard()
            return False

        self.ego_vehicle = vehicle
        return True
