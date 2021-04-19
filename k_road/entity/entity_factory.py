from k_road.entity.curb import Curb
from k_road.entity.entity_type import EntityType
from k_road.entity.lane_marking.lane_marking_white import LaneMarkingWhite
from k_road.entity.lane_marking.lane_marking_white_dashed import LaneMarkingWhiteDashed
from k_road.entity.lane_marking.lane_marking_white_double import LaneMarkingWhiteDouble
from k_road.entity.lane_marking.lane_marking_yellow import LaneMarkingYellow
from k_road.entity.lane_marking.lane_marking_yellow_dashed import LaneMarkingYellowDashed
from k_road.entity.lane_marking.lane_marking_yellow_dashed_left import LaneMarkingYellowDashedLeft
from k_road.entity.lane_marking.lane_marking_yellow_dashed_right import LaneMarkingYellowDashedRight
from k_road.entity.lane_marking.lane_marking_yellow_double import LaneMarkingYellowDouble
from k_road.entity.path import Path
from k_road.entity.target import Target
from k_road.entity.trigger_zone import TriggerZone
from k_road.entity.vehicle.vehicle import Vehicle

entity_factories = {

    EntityType.vehicle: lambda *args, **kwargs: Vehicle(*args, **kwargs),
    # EntityType.pedestrian:                       lambda *args, **kwargs: Pedestrian(*args, **kwargs),
    # EntityType.bicycle:                          lambda *args, **kwargs: Bicycle(*args, **kwargs),
    # EntityType.animal:                           lambda *args, **kwargs: Animal(*args, **kwargs),
    # EntityType.other:                            lambda *args, **kwargs: Other(*args, **kwargs),

    # lanes
    # EntityType.lane:                             lambda *args, **kwargs: Lane(*args, **kwargs),
    # EntityType.bike_lane:                        lambda *args, **kwargs: BikeLane(*args, **kwargs),
    # EntityType.bus_lane:                         lambda *args, **kwargs: BusLane(*args, **kwargs),
    # EntityType.restricted_lane:                  lambda *args, **kwargs: RestrictedLane(*args, **kwargs),
    # EntityType.parking_lane:                     lambda *args, **kwargs: ParkingLane(*args, **kwargs),
    # EntityType.two_way_left_turn:                lambda *args, **kwargs: TwoWayLeftTurn(*args, **kwargs),
    # EntityType.sidewalk:                         lambda *args, **kwargs: Sidewalk(*args, **kwargs),

    # longitudinal markings
    EntityType.lane_marking_yellow: lambda *args, **kwargs: LaneMarkingYellow(*args, **kwargs),
    EntityType.lane_marking_yellow_dashed: lambda *args, **kwargs: LaneMarkingYellowDashed(*args, **kwargs),
    EntityType.lane_marking_yellow_double: lambda *args, **kwargs: LaneMarkingYellowDouble(*args, **kwargs),
    EntityType.lane_marking_yellow_dashed_left: lambda *args, **kwargs: LaneMarkingYellowDashedLeft(*args, **kwargs),
    EntityType.lane_marking_yellow_dashed_right: lambda *args, **kwargs: LaneMarkingYellowDashedRight(*args, **kwargs),
    EntityType.lane_marking_white: lambda *args, **kwargs: LaneMarkingWhite(*args, **kwargs),
    EntityType.lane_marking_white_dashed: lambda *args, **kwargs: LaneMarkingWhiteDashed(*args, **kwargs),
    EntityType.lane_marking_white_double: lambda *args, **kwargs: LaneMarkingWhiteDouble(*args, **kwargs),
    # EntityType.lane_marking_red : lambda *args : LaneMarkingred(*args, **kwargs),
    # EntityType.lane_marking_blue : lambda *args : LaneMarkingblue(*args, **kwargs),

    # transverse markings
    # EntityType.crosswalk:                        lambda *args, **kwargs: Crosswalk(*args, **kwargs),
    # EntityType.stop_line:                        lambda *args, **kwargs: StopLine(*args, **kwargs),
    # EntityType.yield_line:                       lambda *args, **kwargs: YieldLine(*args, **kwargs),
    # EntityType.speed_bump:                       lambda *args, **kwargs: SpeedBump(*args, **kwargs),

    # parking markings
    # parking_space = EntityCategory.dynamic * entity_category_size + 512  # interior of a parking space
    # parking_space_border = EntityCategory.dynamic * entity_category_size + 513

    # non-lane areas
    EntityType.curb: lambda *args, **kwargs: Curb(*args, **kwargs),
    # EntityType.neutral:                          lambda *args, **kwargs: Neutral(*args, **kwargs),

    # signaling
    # EntityType.stop_sign:                        lambda *args, **kwargs: StopSign(*args, **kwargs),
    # EntityType.stop_light:                       lambda *args, **kwargs: StopLight(*args, **kwargs),

    # hints
    EntityType.target: lambda *args, **kwargs: Target(*args, **kwargs),
    EntityType.path: lambda *args, **kwargs: Path(*args, **kwargs),
    # EntityType.intersection:                     lambda *args, **kwargs: Intersection(*args, **kwargs),

    # sensors
    EntityType.trigger_zone: lambda *args, **kwargs: TriggerZone(*args, **kwargs),

}


def make_entity(entity_type: EntityType, *args, **kwargs):
    """
    Factory function that makes a lane marking of the given type.
    """
    return entity_factories[entity_type](*args, **kwargs)
