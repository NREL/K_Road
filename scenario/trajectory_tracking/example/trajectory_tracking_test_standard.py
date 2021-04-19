import multiprocessing
import os
import sys
import tempfile
from datetime import datetime
from pprint import pprint

import ray
from ray import tune
from ray.rllib.agents import Trainer
from ray.tune.logger import UnifiedLogger
from ray.tune.result import DEFAULT_RESULTS_DIR

from command_line_tools import command_line_config
from factored_gym import ActionScaler, factored_gym
from k_road.entity.vehicle.vehicle_dbm3 import VehicleDBM3
from k_road.model.tire_model.Fiala_brush_tire_model import FialaBrushTireModel
from scenario.trajectory_tracking.trajectory_tracking_observer import TrajectoryTrackingObserver
from scenario.trajectory_tracking.trajectory_tracking_process import TrajectoryTrackingProcess
from scenario.trajectory_tracking.trajectory_tracking_rewarder import TrajectoryTrackingRewarder
from scenario.trajectory_tracking.trajectory_tracking_terminator import TrajectoryTrackingTerminator
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from trainer.es_actual import ESActualTrainer

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#
# ray.init(num_gpus=1)

"""
Graphs:


"""


class ThisRoadEnv(factored_gym.FactoredGym):

    def __init__(self, env_config):
        self.spec = lambda: None
        self.spec.max_episode_steps = int(100e3)
        max_distance_error = env_config['max_distance_error']
        max_speed_error = env_config['max_speed_error']
        observation_scaling = 1.0  # 10.0
        action_scaling = 1.0
        super().__init__(
            TrajectoryTrackingProcess(
                max_scan_radius=(10 * max_distance_error),
                **env_config['process']),
            TrajectoryTrackingObserver(observation_scaling, **env_config['observer']),
            TrajectoryTrackingTerminator(
                max_speed_error,
                max_distance_error),
            TrajectoryTrackingRewarder(
                max_speed_error=max_speed_error,
                **env_config['rewarder']),
            [ActionScaler(action_scaling)]
        )


def make_trainer(config) -> Trainer:
    results_path = DEFAULT_RESULTS_DIR
    checkpoint_path = os.path.join(results_path, get_setting(config, 'experiment'))
    checkpoint_path = os.path.join(checkpoint_path, get_setting(config, 'name'))

    logdir_prefix = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")

    print('checkpoint_path: ', checkpoint_path)

    def logger_creator(config):
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        logdir = tempfile.mkdtemp(
            prefix=logdir_prefix, dir=checkpoint_path)
        print('logger creator: ', logdir)
        return UnifiedLogger(config, logdir, None)

    # trainer = rllib.agents.ppo.PPOTrainer(config=config, env=ThisRoadEnv)
    # trainer = rllib.agents.es.ESTrainer(config=config, env=ThisRoadEnv)
    # trainer = ESTweakTrainer(config=config, env=ThisRoadEnv)
    trainer = ESActualTrainer(config=config, env=ThisRoadEnv, logger_creator=logger_creator)

    checkpoint_path = get_setting(config, 'checkpoint')
    if checkpoint_path is not None:
        trainer.restore(checkpoint_path)
    return trainer


def train(config, reporter):
    ego_starting_distance = 600.0
    trainer = make_trainer(config)
    checkpoint_frequency = 5
    max_iters = int(100e3)

    # def set_starting_distance(ego_starting_distance):
    #     trainer.workers.foreach_worker(
    #         lambda ev: ev.foreach_env(
    #             lambda env: env.process.set_starting_distance(ego_starting_distance)))
    #
    # # def set_starting_distance(ego_starting_distance):
    # #     for worker in trainer._workers:
    # #         print(worker)
    # #         worker.env.process.set_starting_distance(ego_starting_distance)
    #
    # set_starting_distance(ego_starting_distance)
    for i in range(max_iters):
        result = trainer.train()
        reporter(**result)

        if i % checkpoint_frequency == 0:
            # checkpoint_path = trainer.logdir
            # checkpoint_path = os.path.join(checkpoint_path, get_setting(config, 'experiment'))
            # checkpoint_path = os.path.join(checkpoint_path, get_setting(config, 'name'))
            # print('ld:', trainer.logdir, 'n:', get_setting(config, 'name'), 'c', get_setting(config, 'checkpoint'),
            # 'p',
            #       checkpoint_path)
            # trainer.save(checkpoint_path)
            checkpoint_path = trainer.save()
            print('saved to checkpoint ', checkpoint_path)


def on_episode_end(info):
    # print(info)
    episode = info['episode']
    # print(info)
    # trainer = info['trainer']
    base_env = info['env']
    # episode.custom_metrics['ego_starting_distance'] = base_env.get_unwrapped()[0].process.ego_starting_distance


def get_setting(config: dict, name: str):
    return config['env_config'][name]


def do_training(config, make_ego_vehicle):
    tune.run(
        train,
        name=get_setting(config, 'experiment'),
        trial_name_creator=tune.function(lambda trial: get_setting(config, 'name')),
        config=config,
        # resources_per_trial={'gpu':1},
    )


def do_testing(config, make_ego_vehicle):
    trainer = make_trainer(config)
    env = ThisRoadEnv(config['env_config'])

    for i in range(10):
        reward = 0.0
        observation = env.reset()
        env.render()
        while True:
            action = trainer.compute_action(observation)
            result = env.step(action)
            observation = result[0]
            reward += result[1]
            env.render()
            if result[2]:
                break
        print('ended with reward', reward)
        env.close()


def make_ego_vehicle_with_tire_model(
        process,
        position,
        velocity,
        yaw,
        angular_velocity,
        tire_model=None,
):
    return VehicleDBM3(
        process,
        (255, 64, 64),
        position,
        velocity,
        yaw=yaw,
        angular_velocity=angular_velocity,
        tire_model=tire_model
    )
    # return VehicleDBM(
    #     process,
    #     (255, 64, 64),
    #     position,
    #     velocity,
    #     yaw=yaw,
    #     angular_velocity=angular_velocity
    #     )


def make_linear_tire_vehicle(*args):
    return make_ego_vehicle_with_tire_model(*args, tire_model=None)


def make_fiala_tire_vehicle(*args):
    return make_ego_vehicle_with_tire_model(*args, tire_model=FialaBrushTireModel())


# def convert_config_value_to_int(key, config):
#     config[key] = int(config[key])
#
# def convert_config_value_to_float(key, config):
#     config[key] = int(config[key])

if not ray.is_initialized():
    ray.init()

num_cpus = multiprocessing.cpu_count()
num_workers = max(1.0, num_cpus - 1)

default_config = {
    # "alpha":              0.1,
    # "noise_stdev":        0.02,
    "episodes_per_batch": 100,  # int((36 - 1) * 3),
    "train_batch_size": int(1000),
    "eval_prob": 0.02,
    "num_workers": num_workers,
    "observation_filter": "MeanStdFilter",
    "noise_size": 250000000,
    "report_length": 10,
    # "num_gpus":            1.0,
    # "num_gpus_per_worker": 1.0 / num_workers,
    # "max_episode_steps": 2000,
    "env_config": {
        'experiment': 'standard',
        'name': 'run',
        'checkpoint': None,
        'mode': 'train',
        'max_distance_error': 2.0,
        'max_speed_error': 5.0,

        'process': {
            'path_generator': 'curriculum_path_generator',
            'tire_model': 'fiala',
        },
        'observer': {
            'num_waypoints': 5,
            'waypoint_spacing': 1.0,
            'use_distances': False,
            'use_target_speeds': True,
            'use_velocity_reference_angle': False,
            'use_alternate_reference_angle': False,
        },
        'rewarder': {
            'distance_error_to_speed_tradeoff': 50.0,
        }
    },
    'model': {
        'conv_filters': None,
        'fcnet_activation': 'relu',
        'fcnet_hiddens': [32, 32, 32, 32, 32],
        # 'vf_share_layers': False,
    },
}

config = command_line_config.parse_config_from_args(sys.argv[1:], default_config)
pprint(config)

env_config = config['env_config']

process_config = env_config['process']
tire_model = process_config['tire_model']
del process_config['tire_model']
make_ego_vehicle = None
if tire_model == 'fiala':
    make_ego_vehicle = make_fiala_tire_vehicle
elif tire_model == 'linear':
    make_ego_vehicle = make_linear_tire_vehicle
else:
    raise Exception('Unknown tire model.')
process_config['make_ego_vehicle'] = tune.function(make_ego_vehicle)

path_generator_setting = process_config['path_generator']
path_generator = None
if path_generator_setting == 'curriculum_path_generator':
    path_generator = TrajectoryTrackingProcess.curriculum_path_generator
elif path_generator_setting == 'fixed_path_generator':
    path_generator = TrajectoryTrackingProcess.fixed_path_generator
elif path_generator_setting == 'sine_path_generator':
    path_generator = TrajectoryTrackingProcess.sine_path_generator
elif path_generator_setting == 'straight_path_generator':
    path_generator = TrajectoryTrackingProcess.straight_path_generator
else:
    raise Exception('Unknown path_generator.')
process_config['path_generator'] = tune.function(path_generator)

print('config:')
pprint(config)

print("Nodes in the Ray cluster:")
pprint(ray.nodes())

mode = get_setting(config, 'mode')
if mode == 'train':
    do_training(config, make_ego_vehicle)
elif mode == 'test':
    do_testing(config, make_ego_vehicle)
else:
    raise Exception('Unknown mode.')

ray.shutdown()
