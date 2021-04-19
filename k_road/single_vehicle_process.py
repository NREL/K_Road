from typing import (
    Any,
    Optional,
)

import gym
import numpy as np

from k_road.entity.vehicle.dynamic_single_body_vehicle import DynamicSingleBodyVehicle
from k_road.k_road_process import KRoadProcess
from k_road.k_road_view import KRoadView


class SingleVehicleProcess(KRoadProcess):
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self,
                 view_follows_ego: bool = False,
                 differential_steering: bool = False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.view_follows_ego: bool = view_follows_ego
        self.ego_vehicle: Optional[DynamicSingleBodyVehicle] = None
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=[2], dtype=np.float32)

    def get_action_space(self):
        return self.action_space

    def reset(self, process: 'SingleVehicleProcess') -> None:
        self.ego_vehicle = None
        super().reset(process)

    def step(self, action) -> Any:
        # acceleration = action[0]
        # if acceleration < 0:
        #     acceleration = acceleration * Constants.max_deceleration
        # else:
        #     acceleration = acceleration * Constants.max_acceleration
        #
        # steer = action[1] * Constants.max_steer_angle
        # self.ego_vehicle.apply_control(self.time_step_length, acceleration, steer)
        # self.ego_vehicle.apply_control(self.time_step_length, acceleration, steer)

        return super().step(action)

    def begin_render(self, process: 'KRoadProcess', view: KRoadView) -> None:
        super().begin_render(process, view)
        if self.view_follows_ego:
            view.set_view_center(self.ego_vehicle.position)

    def render(self, process: 'KRoadProcess', view: KRoadView) -> None:
        super().render(process, view)
