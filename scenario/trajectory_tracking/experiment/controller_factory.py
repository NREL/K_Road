#!/usr/bin/env python3
import os
import sys
from abc import (
    ABC,
    abstractmethod,
)



sys.path.append(str(os.environ['HOME']) + "/cavs-environments")


class UnknownControllerException(Exception):
    pass


class ControllerFactory(ABC):

    def __init__():
        pass

    @abstractmethod
    def create_controller(self):
        pass


class StanleyFactory(ControllerFactory):

    def __init__(self, **kwargs):
        from scenario.trajectory_tracking.controller.stanley import StanleyController

        self.controller = StanleyController(**kwargs)  # all with defaults

    def create_controller(self):
        return self.controller


class PIDFactory(ControllerFactory):

    def __init__(self, **kwargs):
        from scenario.trajectory_tracking.controller.pid_controllers import VehiclePIDController

        self.controller = VehiclePIDController(**kwargs)

    def create_controller(self):
        return self.controller


class PurePursuitFactory(ControllerFactory):

    def __init__(self, **kwargs):
        from scenario.trajectory_tracking.controller.pure_pursuit import PurePursuitController

        self.controller = PurePursuitController(**kwargs)

    def create_controller(self):
        return self.controller


class NullFactory(ControllerFactory):

    def __init__(self, **kwargs):
        from factored_gym import NullController

        self.controller = NullController(action_space=1.0)

    def create_controller(self):
        return self.controller


if __name__ == '__main__':
    controller_dict = {
        'stanley': StanleyFactory,
        'pid': PIDFactory,
        'pure_pursuit': PurePursuitFactory,
        'null': NullFactory,
    }
    try:
        factory = controller_dict['null']
        controller = factory()
        print(controller)
    except:
        raise (UnknownControllerException("controller not found"))
