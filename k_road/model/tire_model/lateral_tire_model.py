from abc import (
    ABC,
    abstractmethod,
)


class LateralTireModel(ABC):

    @abstractmethod
    def get_lateral_force(
            self,
            Fz: float,  # Fz = vertical force on tire (load)
            a: float,  # slip angle
            Fx0: float,  # longitudinal force [N]
            Vc: float,  # wheel contact center velocity magnitude
            k: float,  # longitudinal slip
            g: float = 0.0,  # g (gamma) = camber angle
    ) -> (float, float):
        """
        :returns pure lateral slip force, combined lateral slip force
        """
        pass

    @abstractmethod
    def estimate_stiffness_from_mass_and_spacing(self):
        """
            Sarah: Added this because I need stiffness for
                some model-based controllers
        """
        pass
