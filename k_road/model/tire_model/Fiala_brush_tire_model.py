from math import *

# noinspection PyPep8Naming
from k_road.model.tire_model.lateral_tire_model import LateralTireModel


class FialaBrushTireModel(LateralTireModel):
    """
    Also uses a friction circle to limit lateral forces in high longitudinal force regimes.
    see https://ddl.stanford.edu/sites/g/files/sbiybj9456/f/Zhang_2018_avec.pdf
    see https://www.sciencedirect.com/science/article/pii/S0967066116300831
    see Fiala (1954) and presented by Pacejka (2002)
    """
    """
        Shivam 2018: 
             m = 1,587 kg, Iz = 2,315.3 kg m2, lf = 1.218 m, lr = 1.628 m, and Cα,f = Cα,r = 35,000 N/rad.
        Falcone 2009: Pacejka model (Bakker et al., 1987)
             mu = 0.3; maximum slip angle = 3 deg; mass =  2050; 
             yaw_inertia = 3344 kg/m^2; Cornering stiffness = 
        Lee 2018: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5849315/pdf/pone.0194110.pdf 
             m = 1,270 kg, Iz = 1536.7 kg m2; lf = 1.015 m; lr =  1.895[m]; C = 55,405 N/rad.
    """

    def __init__(
            self,
            mu: float = .9,  # road friction coefficient (dry road)
            # Ca: float = 1.1441e5,  # tire cornering stiffness [N/rad]
            # Ca: float = 675.0,  # tire cornering stiffness [N/rad]
            # Ca: float = 33500.0,  # [N/rad]   tire cornering stiffness [N/rad] coming from estimate
            Ca: float = 1600.0,
            # Ca: float = 160,  # [kN/rad]
    ):

        self.mu: float = mu
        self.Ca: float = Ca

    def estimate_stiffness_from_mass_and_spacing(self, mass, lf, lr):
        return self.Ca

    def get_lateral_force(
            self,
            Fz: float,  # Fz = vertical force on tire (load) [N]
            slip_angle: float,  # slip angle
            Fx0: float,  # longitudinal force [N]
            Vc: float,  # wheel contact center velocity magnitude (unused)
            k: float,  # longitudinal slip (unused)
            g: float = 0.0,  # g (gamma) = camber angle (unused)
    ) -> (float, float):  # [N, N]
        tan_slip_angle = tan(slip_angle)
        if fabs(slip_angle) < atan(3 * self.mu * Fz / self.Ca):
            # print('case 1 ')
            Fy0 = -self.Ca * tan_slip_angle \
                  + ((self.Ca ** 2) / (3 * self.mu * Fz)) * fabs(tan_slip_angle) * tan_slip_angle \
                  - ((self.Ca ** 3) / (27 * (self.mu ** 2) * (Fz ** 2))) * (tan_slip_angle ** 3)
        else:
            print('case 2 ')
            Fy0 = -self.mu * Fz * copysign(1.0, slip_angle)

        # use friction circle to limit Fy
        # Fymax = sqrt(((self.mu * Fz) ** 2) - ((Fx0 / 1000) ** 2))
        # Fymax = sqrt(((self.mu * Fz) ** 2) - (Fx0 ** 2))
        friction_circle_radius_squared = (self.mu * Fz) ** 2
        target_lateral_force_squared = Fx0 ** 2

        Fy = 0.0
        if friction_circle_radius_squared >= target_lateral_force_squared:
            # print('within friction circle ')
            maximum_lateral_force = sqrt(friction_circle_radius_squared - target_lateral_force_squared)
            if fabs(Fy0) > maximum_lateral_force:
                print('friction circle limited {} {}'.format(Fy0, maximum_lateral_force))
            Fy = max(-maximum_lateral_force, min(maximum_lateral_force, Fy0))
        else:
            print('max friction circle ')

        return Fy, Fy0
