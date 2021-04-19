#!/usr/bin/env python3
import os
import sys

from scenario.trajectory_tracking.experiment.experiment_common import make_environment_and_controller

sys.path.append(str(os.environ["HOME"]) + "/cavs-environments")

config = {
    'name': 'test', 'group': 'test', 'max_run_length': 10000000, 'num_cpus': None, 'checkpoint': None,
    'controller': 'stanley', 'environment': {
        'max_distance_error': 5.0, 'max_speed_error': 10.0, 'controller': {}, 'process': {
            'path_generator': 'carla_json_generator', 'path_generator_config': {}, 'tire_model': 'fiala'
        }, 'observer': {
            'num_waypoints': 10, 'waypoint_spacing': 1.0, 'use_angles': True, 'use_distances': True,
            'use_relative_distances': False, 'use_target_speeds': False,
            'use_velocity_reference_angle': False, 'use_alternate_reference_angle': False,
            'use_relative_angles': False, 'space_waypoints_with_actual_speed': False, 'mirror': False
        }, 'rewarder': {'distance_error_to_speed_tradeoff': 20.0}, 'terminator': {}
    }, 'rllib': {
        'episodes_per_batch': 144, 'train_batch_size': 1000, 'eval_prob': 0.02, 'num_workers': 7,
        'observation_filter': 'MeanStdFilter', 'noise_size': 250000000, 'report_length': 10, 'env_config': {},
        'model': {
            'conv_filters': None, 'fcnet_activation': 'relu', 'fcnet_hiddens': [128, 128, 128, 128, 128, 128]
        }
    }, 'num_runs': 1, 'render': True
}

env, trainer = make_environment_and_controller(config, None)

while True:
    action = trainer.compute_action(env.process)
    observation, reward, is_terminal, extra = env.step(action)
    env.render()
