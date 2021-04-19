
# path_generators = [
#     'curriculum_angled_path_factory',
#     'circle_path_factory',
#     'figure_eight_generator',
#     # 'carla_json_generator',
#     'double_lane_change_generator',
#     'straight_variable_speed_pulse_generator',
#     'hairpin_turn_generator',
#     'hairpin_turn_flat_genera tor',
#     'right_turn_generator',
#     'right_turn_flat_generator',
#     'snider_2009_track_generator',
#     ]

# parse_rl_controller_checkpoint
import sys

from scenario.trajectory_tracking.experiment.experiment_common import common_default_config
from scenario.trajectory_tracking.experiment.trajectory_tracking_recorder import parse_rl_controller_checkpoint, \
    run_checkpoint_experiment

path_generator = sys.argv[1]
checkpoint = sys.argv[2]

print('running checkpoint experiment: path generator: {},  checkpoint: {}.'.format(
    path_generator, checkpoint))

checkpoint_info = parse_rl_controller_checkpoint(checkpoint)
run_checkpoint_experiment(common_default_config, checkpoint_info, path_generator, False)

#
# for path_generator in path_generators:
#     print('path generator: {}, checkpoint: {}.'.format(path_generator, checkpoint))
#     run_checkpoint_experiment(common_default_config, checkpoint_info, path_generator, False)


# checkpoint_path = sys.argv[1]
# rl_config_dictionary = parse_json_rl_controllers(checkpoint_path)
# run_checkpoint_experiments(
#     rl_controller_dictionary=rl_config_dictionary,
#     path_generators=path_generators,
#     render=False,
#     default_config=common_default_config)
