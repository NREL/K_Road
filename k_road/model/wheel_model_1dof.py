class WheelModel1DOF:
    '''
    A simple wheel model with only the angular velocity as state
    '''

    def __init__(
            self,
            radius: float = .313,  # [m] includes tire, etc
            mass: float = 9.3,  # [kg] includes tire, etc
    ):
        self.radius: float = radius
        self.mass: float = mass
        self.Iz: float = self.mass * self.radius ** 2  # rotational inertia about axel

    def calc_derivatives(
            self,
            angular_velocity,
            torque,
    ):
        angular_acceleration = torque / self.Iz
        return angular_acceleration
