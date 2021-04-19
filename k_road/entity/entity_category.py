from enum import IntEnum


class EntityCategory(IntEnum):
    """
    Categorizes entities into useful categories. Also used as pymunk collision categories, so these must all be
    powers of two, from 2^0 to 2^31.
    """
    null = 0
    dynamic = 2
    lane = 2 * dynamic
    lane_marking = 2 * lane
    transverse_marking = 2 * lane_marking
    off_road = 2 * transverse_marking
    neutral = 2 * off_road
    signal = 2 * neutral
    hint = 2 * signal

    sensor = 2 * hint
    dynamic_sensor = 2 * sensor
    lane_sensor = 2 * dynamic_sensor
    lane_marking_sensor = 2 * lane_sensor
    transverse_marking_sensor = 2 * lane_marking_sensor
    off_road_sensor = 2 * transverse_marking_sensor
    signal_sensor = 2 * off_road_sensor
    hint_sensor = 2 * signal_sensor


"""
CollisionMasks defines what entity categories can collide with an entity category
"""
CollisionMasks = {
    EntityCategory.dynamic:
        EntityCategory.dynamic |
        EntityCategory.off_road |
        EntityCategory.sensor |
        EntityCategory.dynamic_sensor,
    EntityCategory.lane:
        EntityCategory.sensor |
        EntityCategory.lane_sensor,
    EntityCategory.lane_marking:
        EntityCategory.sensor |
        EntityCategory.lane_marking_sensor,
    EntityCategory.transverse_marking:
        EntityCategory.sensor |
        EntityCategory.transverse_marking_sensor,
    EntityCategory.off_road:
        EntityCategory.dynamic |
        EntityCategory.sensor |
        EntityCategory.off_road_sensor,
    EntityCategory.neutral:
        EntityCategory.sensor,
    EntityCategory.signal:
        EntityCategory.sensor |
        EntityCategory.signal_sensor,
    EntityCategory.hint:
        EntityCategory.hint_sensor | EntityCategory.sensor,

    EntityCategory.sensor:
        EntityCategory.dynamic |
        EntityCategory.lane |
        EntityCategory.transverse_marking |
        EntityCategory.off_road |
        EntityCategory.neutral |
        EntityCategory.signal |
        EntityCategory.hint,

    EntityCategory.dynamic_sensor: EntityCategory.dynamic,

    EntityCategory.lane_sensor: EntityCategory.lane,
    EntityCategory.lane_marking_sensor: EntityCategory.lane_marking,
    EntityCategory.transverse_marking_sensor: EntityCategory.transverse_marking,
    EntityCategory.off_road_sensor: EntityCategory.off_road,
    EntityCategory.signal_sensor: EntityCategory.signal,
    EntityCategory.hint_sensor: EntityCategory.hint
}
