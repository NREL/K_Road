# Idea here is this is a minimal script that can live on /scratch and drive a run.
# This will be used as a template for fiddling with parameters below and submitting separate jobs for each set

import numpy as np
from cavs_environments.vehicle.deep_road.deep_road_constants import DeepRoadConstants

from scenario import do_train

if __name__ == "__main__":
    inp = {}
    rewarder_config = {'collision_bonus': -10, 'distance_bonus': 3, 'completion_bonus': 10, 'step_bonus': 0.1}
    process_config = {"min_plan_length": 30, "min_step_dist": 0.4 * DeepRoadConstants.car_length,
                      "max_step_dist": 3 * DeepRoadConstants.car_length, "max_step_angle": 30 * np.pi / 180.0}
    inp['name'] = 'path_planning'
    inp['scenario'] = 'avoid_bots'
    inp['rewarder_config'] = rewarder_config
    inp['process_config'] = process_config
    # inp dict provides overrides for the default config
    # so off we go...
    do_train(inp)
