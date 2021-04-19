import os
from pprint import pprint

from scenario.trajectory_tracking.experiment.trajectory_tracking_recorder import parse_json_rl_controllers

path_generators = [
    'curriculum_angled_path_factory',
    'circle_path_factory',
    'figure_eight_generator',
    # 'carla_json_generator',
    'double_lane_change_generator',
    'straight_variable_speed_pulse_generator',
    'hairpin_turn_generator',
    'hairpin_turn_flat_genera tor',
    'right_turn_generator',
    'right_turn_flat_generator',
    'snider_2009_track_generator',
    ]


# path_generator = sys.argv[1]
# checkpoint = sys.argv[1]
#
# checkpoint_info = parse_rl_controller_checkpoint(checkpoint)
#
#
# for path_generator in path_generators:
#     print('path generator: {}, checkpoint: {}.'.format(path_generator, checkpoint))
#     run_checkpoint_experiment(common_default_config, checkpoint_info, path_generator, False)
#

checkpoint_path = os.path.join(os.getcwd(), "apr10_2021_controllers")
rl_config_dictionary = parse_json_rl_controllers(checkpoint_path)
# rl_config_dictionary = parse_json_rl_controllers(checkpoint_path)

print('rl_config_dictionary:')
pprint(rl_config_dictionary)

for info in rl_config_dictionary.values():
    for path_generator in path_generators:
        # config = info['config']
        # name = config['name']
        file = info['file']
        command = 'python run_checkpoint_experiment.py {} {}'.format(path_generator, file)
        print(command)
        os.system(command)
        # run_checkpoint_experiment(common_default_config, info, path_generator, False)
