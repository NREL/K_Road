from enum import IntEnum


class EntityGroup(IntEnum):
    """
    Groups entities into useful categories. Also used as pymunk collision categories, so these must all be
    powers of two, from 2^0 to 2^31.
    """

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
CollisionMasks defines what entity groups can collide with an entity group
"""
CollisionMasks = {
    EntityGroup.dynamic:
        EntityGroup.dynamic |
        EntityGroup.off_road |
        EntityGroup.sensor |
        EntityGroup.dynamic_sensor,
    EntityGroup.lane:
        EntityGroup.sensor |
        EntityGroup.lane_sensor,
    EntityGroup.lane_marking:
        EntityGroup.sensor |
        EntityGroup.lane_marking_sensor,
    EntityGroup.transverse_marking:
        EntityGroup.sensor |
        EntityGroup.transverse_marking_sensor,
    EntityGroup.off_road:
        EntityGroup.dynamic |
        EntityGroup.sensor |
        EntityGroup.off_road_sensor,
    EntityGroup.neutral:
        EntityGroup.sensor,
    EntityGroup.signal:
        EntityGroup.sensor |
        EntityGroup.signal_sensor,
    EntityGroup.hint:
        EntityGroup.hint_sensor,

    EntityGroup.sensor:
        EntityGroup.dynamic |
        EntityGroup.lane |
        EntityGroup.transverse_marking |
        EntityGroup.off_road |
        EntityGroup.neutral |
        EntityGroup.signal,

    EntityGroup.dynamic_sensor: EntityGroup.dynamic,

    EntityGroup.lane_sensor: EntityGroup.lane,
    EntityGroup.lane_marking_sensor: EntityGroup.lane_marking,
    EntityGroup.transverse_marking_sensor: EntityGroup.transverse_marking,
    EntityGroup.off_road_sensor: EntityGroup.off_road,
    EntityGroup.signal_sensor: EntityGroup.signal,
    EntityGroup.hint_sensor: EntityGroup.hint
}
