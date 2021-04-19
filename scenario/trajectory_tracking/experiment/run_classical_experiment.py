import sys
from pprint import pprint

from command_line_tools.command_line_config import merge_configs
from command_line_tools.run_tools import setup_run
from scenario.trajectory_tracking.experiment.experiment_common import common_default_config
from scenario.trajectory_tracking.experiment.trajectory_tracking_recorder import run

path_generator = sys.argv[1]
controller = sys.argv[2]


def run_one(path_generator, controller):
    print('path generator: {}, controller: {}.'.format(path_generator, controller))
    
    config = merge_configs(common_default_config, {
        'num_runs':    1,
        'render':      True,
        'controller':  controller,
        'name':        controller + "_" + path_generator,
        'environment': {
            'max_distance_error': 25.0,
            'process':            {
                'path_generator': path_generator,
                }
            }
        })
    print('merged:')
    pprint(config)
    print('running:')
    config, run_prefix = setup_run(config, use_command_line=False)
    run(config, run_prefix, 0)
    print('complete.')

run_one(path_generator, controller)
