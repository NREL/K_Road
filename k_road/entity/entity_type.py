from enum import IntEnum

from k_road.entity.entity_category import EntityCategory

entity_category_size = 64


class EntityType(IntEnum):
    null = 0

    # dynamic entities
    vehicle = EntityCategory.dynamic * entity_category_size + 0
    pedestrian = EntityCategory.dynamic * entity_category_size + 1
    bicycle = EntityCategory.dynamic * entity_category_size + 2
    animal = EntityCategory.dynamic * entity_category_size + 3
    other = EntityCategory.dynamic * entity_category_size + 4

    # lanes
    lane = EntityCategory.lane * entity_category_size + 0  # the interior of a lane
    bike_lane = EntityCategory.lane * entity_category_size + 1
    bus_lane = EntityCategory.lane * entity_category_size + 2
    restricted_lane = EntityCategory.lane * entity_category_size + 3
    parking_lane = EntityCategory.lane * entity_category_size + 4
    two_way_left_turn = EntityCategory.lane * entity_category_size + 5
    sidewalk = EntityCategory.lane * entity_category_size + 6

    # longitudinal markings
    lane_marking_yellow = EntityCategory.lane_marking * entity_category_size + 0
    lane_marking_yellow_dashed = EntityCategory.lane_marking * entity_category_size + 1
    lane_marking_yellow_double = EntityCategory.lane_marking * entity_category_size + 2
    lane_marking_yellow_dashed_left = EntityCategory.lane_marking * entity_category_size + 3
    lane_marking_yellow_dashed_right = EntityCategory.lane_marking * entity_category_size + 4

    lane_marking_white = EntityCategory.lane_marking * entity_category_size + 5
    lane_marking_white_dashed = EntityCategory.lane_marking * entity_category_size + 6
    lane_marking_white_double = EntityCategory.lane_marking * entity_category_size + 7

    lane_marking_red = EntityCategory.lane_marking * entity_category_size + 8
    lane_marking_blue = EntityCategory.lane_marking * entity_category_size + 9
    # possibly: dotted markings as well as dashed and solid, and possibly virtual markings?

    # transverse markings
    crosswalk = EntityCategory.transverse_marking * entity_category_size + 0
    stop_line = EntityCategory.transverse_marking * entity_category_size + 1
    yield_line = EntityCategory.transverse_marking * entity_category_size + 2
    speed_bump = EntityCategory.transverse_marking * entity_category_size + 3  # is this a marking?

    # parking markings
    # parking_space = EntityCategory.dynamic * entity_category_size + 512  # interior of a parking space
    # parking_space_border = EntityCategory.dynamic * entity_category_size + 513

    # non-lane areas
    curb = EntityCategory.off_road * entity_category_size + 0  # non-drivable regions

    neutral = EntityCategory.neutral * entity_category_size + 0

    # signaling
    stop_sign = EntityCategory.signal * entity_category_size + 0
    stop_light = EntityCategory.signal * entity_category_size + 1

    # hints
    target = EntityCategory.hint * entity_category_size + 0
    path = EntityCategory.hint * entity_category_size + 1
    intersection = EntityCategory.hint * entity_category_size + 2

    # sensors
    trigger_zone = EntityCategory.dynamic_sensor * entity_category_size + 3

    # rail crossings, etc?

    @property
    def category(self) -> EntityCategory:
        return EntityCategory(int(int(self) / entity_category_size))
