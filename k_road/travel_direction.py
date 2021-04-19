from enum import IntEnum


class TravelDirection(IntEnum):
    """
    For specifying travel directions along lanes
    """
    none = 0
    forward = 1
    backward = 2
    bidirectional = 3
    bidirectional_left_turn = 4
