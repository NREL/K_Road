import json

import numpy as np
from cavs_environments.vehicle.deep_road.deep_road_constants import DeepRoadConstants

from scenario import do_train


def first_test():
    inp = {}
    rewarder_config = {'collision_bonus': -10, 'distance_bonus': 1, 'completion_bonus': 100, 'step_bonus': 0.1}
    process_config = {"min_plan_length": 30, "min_step_dist": 0.2 * DeepRoadConstants.car_length,
                      "max_step_dist": 2 * DeepRoadConstants.car_length, "max_step_angle": 20 * np.pi / 180.0}
    inp['name'] = 'path_planning'
    inp['scenario'] = 'avoid_bots'
    inp['rewarder_config'] = rewarder_config
    inp['process_config'] = process_config
    # inp dict provides overrides for the default config
    # so off we go...


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inpfile",
        type=str,
        default=None)
    args = parser.parse_args()

    if args.inpfile != None:
        inp = json.load(open(args.inpfile))
    else:
        inp = {}

    do_train(inp)
