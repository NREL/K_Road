import math


class Constants():
    # scaling constants (mks units)
    lane_width = 3.7  # US interstate highway standard is 12ft (3.7m)

    curb_width = .15  # typical curb width
    gutter_width = .2  # typical gutter width
    # curb_width = lane_width * 10  # typical curb width

    # default_acceleration = 4.47  # 10mph change in speed per second
    max_acceleration = 13.4 / 2  # 15 mph / s  ( 5 and 2 recommended...)
    max_deceleration = -26.8  # 30 mph / s
    # max_acceleration = 2
    # max_deceleration = 5

    max_jerk = 33.57
    min_jerk = -2 * 33.57

    max_steer_rate = (30 / 360) * 2 * math.pi
    max_steer_angle = (60 / 360) * 2 * math.pi  # 60 or 45?

    lane_change_time = 4

    max_speed = 50.0  # ~112mph
