from math import *

# noinspection PyPep8Naming
from k_road.model.tire_model.lateral_tire_model import LateralTireModel


class LinearTireModel(LateralTireModel):
    """
    Simply uses a linear model with an intercept of 0 and slope of Ca and a friction circle at
    """

    def __init__(
            self,
            mu: float = .9,  # road friction coefficient
            # Ca: float = 1.1441e5,  # tire cornering stiffness [N/rad]
            Ca: float = 150e3,  # tire cornering stiffness [N/rad]
    ):
        self.mu: float = mu
        self.Ca: float = Ca

    def get_lateral_force(
            self,
            Fz: float,  # Fz = vertical force on tire (load)
            a: float,  # slip angle
            Fx0: float,  # longitudinal force [N]
            Vc: float,  # wheel contact center velocity magnitude
            k: float,  # longitudinal slip
            g: float = 0.0,  # g (gamma) = camber angle
    ) -> (float, float):
        Fy0 = -self.Ca * a

        # use friction circle to limit Fy
        Fymax = sqrt((self.mu * Fz) ** 2 + Fx0 ** 2)
        Fy = max(-Fymax, min(Fymax, Fy0))

        return Fy, Fy0

    def estimate_stiffness_from_mass_and_spacing(
            self,
            mass: float,  # [kg]
            front_wheel_spacing: float,  # [m]
            rear_wheel_spacing: float,  # [m]
    ) -> float:
        """
        see W. Sienel, "Estimation of the tire cornering stiffness and its application to active car steering,
        " Proceedings of the 36th IEEE Conference on Decision and Control, San Diego, CA, USA, 1997, pp. 4744-4749
        vol.5.
            doi: 10.1109/CDC.1997.649759
            URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=649759&isnumber=14185

        see also used in FORDS: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2017/EECS-2017-102.pdf
        """
        self.Ca = self.mu * mass * (rear_wheel_spacing / (front_wheel_spacing + rear_wheel_spacing))
        return self.Ca
