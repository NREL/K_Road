import math


class IntelligentDriverModel:
    """
    see https://en.wikipedia.org/wiki/Intelligent_driver_model
    see Martin Treiber, Ansgar Hennecke, Dirk Helbing,
        'Congested Traffic States in Empirical Observations and Microscopic Simulations'
        https://arxiv.org/pdf/cond-mat/0002177.pdf
    using default parameters for motorways as listed in
        https://pdfs.semanticscholar.org/02e8/0e11f7a0bbd53b83125c0a20880e92119483.pdf
    """

    def __init__(self,
                 safe_time_headway: float = 1.2,
                 max_acceleration: float = 1.25,
                 comfortable_deceleration: float = 1.67,
                 acceleration_exponent: float = 4.0,
                 min_distance: float = 3.0):
        self.safe_time_headway: float = safe_time_headway
        self.max_acceleration: float = max_acceleration
        self.comfortable_deceleration: float = comfortable_deceleration
        self.acceleration_exponent: float = acceleration_exponent
        self.min_distance: float = min_distance

    def get_acceleration(self,
                         desired_speed: float,
                         speed: float,
                         obstacle_speed: float,
                         obstacle_distance: float
                         ) -> float:
        approaching_rate = speed - obstacle_speed

        if obstacle_distance < .1:
            return -approaching_rate * .02

        free_road_term = 1 - math.pow(speed / desired_speed, self.acceleration_exponent)
        interaction_term = (self.min_distance + speed * self.safe_time_headway) / obstacle_distance + \
                           (speed * approaching_rate) / \
                           (2 * math.sqrt(self.max_acceleration * self.comfortable_deceleration) * obstacle_distance)
        interaction_term = interaction_term * interaction_term
        return self.max_acceleration * (free_road_term - interaction_term)

    def get_free_acceleration(self, desired_speed: float, speed: float) -> float:
        free_road_term = 1 - math.pow(speed / desired_speed, self.acceleration_exponent)
        return self.max_acceleration * free_road_term
