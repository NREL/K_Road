import multiprocessing
import os
import tempfile
from copy import deepcopy
from datetime import datetime

import ray
from pymunk import Vec2d
from ray.tune.logger import UnifiedLogger

from k_road.entity.vehicle.DynamicSBVActionTransform import DynamicSBVActionTransform
from k_road.entity.vehicle.vehicle_matlab_3dof import VehicleMatlab3DOF
from k_road.model.tire_model.Fiala_brush_tire_model import FialaBrushTireModel
from scenario.trajectory_tracking import trajectory_generators
from scenario.trajectory_tracking.controller.pid_controllers import VehiclePIDController
from scenario.trajectory_tracking.controller.pure_pursuit import PurePursuitController
from scenario.trajectory_tracking.controller.scheduled_controller import ScheduledController
from scenario.trajectory_tracking.controller.stanley import StanleyController
from scenario.trajectory_tracking.experiment.controller_factory import UnknownControllerException
from scenario.trajectory_tracking.trajectory_tracking_observer import TrajectoryTrackingObserver
from scenario.trajectory_tracking.trajectory_tracking_process import TrajectoryTrackingProcess
from scenario.trajectory_tracking.trajectory_tracking_rewarder import TrajectoryTrackingRewarder
from scenario.trajectory_tracking.trajectory_tracking_terminator import TrajectoryTrackingTerminator
from factored_gym import NullController
from factored_gym.action_transform.additive_action_transform import AdditiveActionTransform
from factored_gym.factored_gym import FactoredGym
from trainer.coordinated_dps_trainer import CoordinatedDPSTrainer
from trainer.es_actual import ESActualTrainer
from trainer.es_co_trainer import ESCOTrainer

num_cpus = multiprocessing.cpu_count()
num_workers = max(1.0, num_cpus - 1)

common_default_config = {
    'name':           'test',
    'group':          'test',
    'max_run_length': 10000000,
    'num_cpus':       None,
    'checkpoint':     None,
    'controller':     None,
    'trainer':        None,
    'environment':    {
        # 'max_cross_track_error': 3.0,
        'max_cross_track_error': 10.0,
        'max_speed_error':       50.0,
        'cross_track_position':  0,
        # 'controller':            {},
        'process':               {
            'path_generator':        'curriculum_path_factory',
            'path_generator_config': {},
            'tire_model':            'fiala',
            # 'time_dilation':         1,
            'time_dilation':         20,
            'scale_actions':         False,
            },
        'observer':              {
            'num_waypoints':                     10,
            'waypoint_spacing':                  1.0,
            'use_angles':                        True,
            'use_distances':                     True,
            'use_relative_distances':            False,
            'use_target_speeds':                 False,
            'use_velocity_reference_angle':      False,
            'use_alternate_reference_angle':     False,
            'use_relative_angles':               False,
            'space_waypoints_with_actual_speed': False,
            'mirror':                            False,
            },
        'rewarder':              {
            'speed_tradeoff':                   .0036,
            'steering_control_effort_tradeoff': 3.65,
            },
        'terminator':            {
            
            },
        'controller':            {
            # 'Q':             [[10., 0., 0., 0.],
            #                   [0., 10., 0., 0.],
            #                   [0., 0., 200., 0.],
            #                   [0., 0., 0., 10.]],
            # 'R':             [[1., 0.],
            #                   [0., 1.]],
            # 'R_bar':         [[1., 0.],
            #                   [0., 1.0]],
            # 'umin':          [-1.5, -np.deg2rad(37.)],
            # 'umax':          [1., np.deg2rad(37.)],
            # 'delumin':       [-3., -np.deg2rad(10.)],
            # 'delumax':       [1.5, np.deg2rad(10.)],
            # 'N':             5,
            # 'Nc':            3,
            # 'output_states': [0, 1, 1, 0, 1, 1],
            # |X| |Y| x_dot y_dot yaw yaw_dot
            },
        },
    'rllib':
                      {
                          # "alpha":              0.1,
                          # "noise_stdev":        0.02,
                          # "candidates_per_iteration": 720,  # int((36 - 1) * 3),
                          # 'num_evals_per_iteration': 4,
                          # 'request_interleaving':    2,
                          # 'alpha':                    0.5,
                          # 'theta_decay':              0.0,
                          # 'noise_stdev':              0.02,
                          # "train_batch_size":         int(1000),
                          # "eval_prob":                0.02,
                          "num_workers":        num_workers,
                          "observation_filter": 'NoFilter',
                          # "observation_filter": "MeanStdFilter",
                          # "noise_size":               250000000,
                          # "report_length":            10,
                          "num_gpus":           0.0,
                          # "num_gpus_per_worker": 1.0 / num_workers,
                          # "max_episode_steps": 2000,
                          "env_config":         {
                              },
                          'model':              {
                              'conv_filters':     None,
                              'fcnet_activation': 'relu',
                              'fcnet_hiddens':    [128, 128, 128, 128, 128, 128],
                              # 'vf_share_layers': False,
                              },
                          },
    # 'optimizer':
    #                   {
    #                       # 'name': 'ESEVRLCoordinatedOptimizer',
    #                       # 'name': 'ESBalancedRLCoordinatedOptimizer',
    #                       # 'name': 'NCEMCoordinatedOptimizer',
    #                       # 'name': 'CanonicalESCoordinatedOptimizer',
    #                       # 'name': 'WCEMCoordinatedOptimizer',
    #                       # 'name': 'CEMCoordinatedOptimizer',
    #                       # 'name':                     'ESRLCoordinatedOptimizer',
    #                       'name':                     'ESBestCoordinatedOptimizer',
    #
    #                       # "alpha":              0.1,
    #                       # "noise_stdev":        0.02,
    #                       "candidates_per_iteration": 720,  # int((36 - 1) * 3),
    #                       # 'num_evals_per_iteration':  4,
    #                       # 'request_interleaving':     2,
    #                       # 'alpha':                    0.5,
    #                       # 'theta_decay':              0.0,
    #                       # 'noise_stdev':              0.02,
    #                       # "train_batch_size":         int(1000),
    #                       # "eval_prob":                0.02,
    #                       # "num_workers":              num_workers,
    #                       # "observation_filter":       "MeanStdFilter",
    #                       # "noise_size":               250000000,
    #                       # "report_length":            10,
    #                       # "num_gpus":            1.0,
    #                       # "num_gpus_per_worker": 1.0 / num_workers,
    #                       # "max_episode_steps": 2000,
    #                       # "env_config":               {
    #                       #     },
    #                       # 'model':                    {
    #                       #     'conv_filters':     None,
    #                       #     'fcnet_activation': 'relu',
    #                       #     'fcnet_hiddens':    [128, 128, 128, 128, 128, 128],
    #                       #     # 'vf_share_layers': False,
    #                       #     },
    #                       # }
    #                       }
    }


def make_ego_vehicle_with_tire_model(
        process,
        position,
        velocity,
        yaw,
        angular_velocity,
        tire_model=None,
        ):
    # TODO: Fix this so vehicle model is selectable
    # return VehicleDBM3(
    #     process,
    #     (255, 64, 64),
    #     position,
    #     velocity,
    #     yaw=yaw,
    #     angular_velocity=angular_velocity,
    #     tire_model=tire_model
    #     )
    return VehicleMatlab3DOF(
        process,
        (255, 64, 64),
        position,
        velocity,
        yaw=yaw,
        angular_velocity=angular_velocity,
        tire_model=tire_model
    )


def vector_to_tuple(v: Vec2d) -> (float, float):
    return (v.x, v.y)


def list_of_vectors_to_list_of_tuples(vecs: [Vec2d]) -> [(float, float)]:
    return [vector_to_tuple(v) for v in vecs]


def setup_environment(config: {}) -> {}:
    config = deepcopy(config)
    environment_config = config['environment']
    process_config = environment_config['process']
    tire_model = process_config['tire_model']
    
    del process_config['tire_model']
    make_ego_vehicle = None
    if tire_model == 'fiala':
        def make_fiala_tire_vehicle(*args):
            return make_ego_vehicle_with_tire_model(*args, tire_model=FialaBrushTireModel())
        
        make_ego_vehicle = make_fiala_tire_vehicle
    
    elif tire_model == 'linear':
        def make_linear_tire_vehicle(*args):
            return make_ego_vehicle_with_tire_model(*args, tire_model=None)
        
        make_ego_vehicle = make_linear_tire_vehicle
    
    else:
        raise Exception('Unknown tire model.')
    # process_config['make_ego_vehicle'] = tune.function(make_ego_vehicle)
    process_config['make_ego_vehicle'] = make_ego_vehicle
    
    path_generator_setting = process_config['path_generator']
    path_generator_config = process_config['path_generator_config']
    del process_config['path_generator_config']
    
    # Sarah TODO: I think we can use dynamic instantiation for what follows:
    path_generator = None
    if path_generator_setting == 'curriculum_path_generator':
        path_generator = trajectory_generators.curriculum_trajectory_generator
    elif path_generator_setting == 'curriculum_path_factory':
        path_generator = trajectory_generators.curriculum_trajectory_factory(**path_generator_config)
    elif path_generator_setting == 'curriculum_curved_path_factory':
        path_generator = trajectory_generators.curriculum_curved_trajectory_factory(**path_generator_config)
    elif path_generator_setting == 'curriculum_angled_path_factory':
        path_generator = trajectory_generators.curriculum_angled_trajectory_factory(**path_generator_config)
    elif path_generator_setting == 'fixed_path_generator':
        path_generator = trajectory_generators.fixed_trajectory_generator
    elif path_generator_setting == 'sine_path_generator':
        path_generator = trajectory_generators.sine_trajectory_generator
    elif path_generator_setting == 'sine_path_factory':
        path_generator = trajectory_generators.sine_trajectory_factory(**path_generator_config)
    elif path_generator_setting == 'straight_path_generator':
        path_generator = trajectory_generators.straight_trajectory_factory(**path_generator_config)
    elif path_generator_setting == 'circle_path_factory':
        path_generator = trajectory_generators.circle_trajectory_factory(**path_generator_config)
    elif path_generator_setting == 'figure_eight_generator':
        path_generator = trajectory_generators.figure_eight_generator
    elif path_generator_setting == 'square_path_generator':
        path_generator = trajectory_generators.square_trajectory_generator
    elif path_generator_setting == 'carla_json_generator':
        path_generator = trajectory_generators.carla_json_generator
    elif path_generator_setting == 'straight_variable_speed_generator':
        path_generator = trajectory_generators.straight_variable_speed_generator
    elif path_generator_setting == 'left_lane_change_generator':
        path_generator = trajectory_generators.left_lane_change_generator
    elif path_generator_setting == 'right_lane_change_generator':
        path_generator = trajectory_generators.right_lane_change_generator
    elif path_generator_setting == 'snider_2009_track_generator':
        path_generator = trajectory_generators.snider_2009_track_generator
    elif path_generator_setting == 'double_lane_change_generator':
        path_generator = trajectory_generators.falcone_2007_generator
    elif path_generator_setting == 'straight_variable_speed_pulse_generator':
        path_generator = trajectory_generators.straight_variable_speed_pulse_generator
    elif path_generator_setting == 'hairpin_turn_generator':
        path_generator = trajectory_generators.hairpin_turn_generator
    elif path_generator_setting == 'hairpin_turn_flat_generator':
        path_generator = trajectory_generators.hairpin_turn_flat_generator()
    elif path_generator_setting == 'right_turn_generator':
        path_generator = trajectory_generators.right_turn_generator
    elif path_generator_setting == 'right_turn_flat_generator':
        path_generator = trajectory_generators.right_turn_flat_generator
    else:
        raise Exception('Unknown path_generator.')
    # process_config['path_generator'] = tune.function(path_generator)
    process_config['path_generator'] = path_generator
    
    controller_class = None
    controller_setting = config['controller']
    if controller_setting is not None:
        controller_dict = {
            'stanley':      StanleyController,
            'pid':          VehiclePIDController,
            'pure_pursuit': PurePursuitController,
            'scheduled':    ScheduledController,
            }
        try:
            controller_class = controller_dict[controller_setting]
        except:
            raise (UnknownControllerException("Controller not found"))
    
    class ThisRoadEnv(FactoredGym):
        
        def __init__(self, env_config):
            self.spec = lambda: None
            self.spec.max_episode_steps = int(100e3)
            max_cross_track_error = environment_config['max_cross_track_error']
            max_speed_error = environment_config['max_speed_error']
            cross_track_position = environment_config['cross_track_position']
            observation_scaling = 1.0  # 10.0
            action_transforms = [DynamicSBVActionTransform()]
            
            process = TrajectoryTrackingProcess(
                # max_scan_radius=(1 * max_cross_track_error + 20),
                max_cross_track_error=max_cross_track_error,
                max_speed_error=max_speed_error,
                cross_track_position=cross_track_position,
                **environment_config['process'],
                )
            
            additive_transform = None
            if controller_class is not None:
                controller = controller_class(**environment_config['controller'])
                additive_transform = AdditiveActionTransform(1.0, 1.0, controller)
                action_transforms.append(additive_transform)
            
            super().__init__(
                process,
                TrajectoryTrackingObserver(
                    observation_scaling,
                    additive_transform=additive_transform,
                    **environment_config['observer']),
                TrajectoryTrackingTerminator(
                    **environment_config['terminator'],
                    ),
                # TimeLimitTerminator(1600*.05),
                TrajectoryTrackingRewarder(
                    **environment_config['rewarder'],
                    ),
                action_transforms
                )
    
    return config, ThisRoadEnv


def make_rllib_config(config):
    rllib_config = config['rllib']
    # nest config in rllib_config to pass it through to rllib function & workers
    del config['rllib']
    rllib_config['env_config']['config'] = config
    return rllib_config


def make_environment_and_controller(config, rllib_config):
    if config is None:
        config = rllib_config['env_config']['config']  # unwrap config from rllib config
    config, this_env = setup_environment(config)
    
    if rllib_config is None:
        rllib_config = make_rllib_config(config)
    environment = this_env({})
    
    results_path = ray.tune.result.DEFAULT_RESULTS_DIR
    checkpoint_path = results_path
    logdir_prefix = config['group'] + '_' + config['name'] + '_' + datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    
    def logger_creator(config):
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        logdir = tempfile.mkdtemp(
            prefix=logdir_prefix, dir=checkpoint_path)
        print('logger creator: ', logdir)
        return UnifiedLogger(config, logdir, None)
    
    environment_config = config['environment']
    trainer_setting = config['trainer']
    trainer = None
    if trainer_setting is None:
        trainer = NullController(environment.process.get_action_space())
    else:
        if not ray.is_initialized():
            ray.init()
        if trainer_setting == 'ESActualTrainer':
            trainer = ESActualTrainer(config=rllib_config, env=this_env, logger_creator=logger_creator)
        elif trainer_setting == 'ESCOTrainer':
            trainer = ESCOTrainer(config=rllib_config, env=this_env, logger_creator=logger_creator)
        elif trainer_setting == 'CoordinatedDPSTrainer':
            trainer = CoordinatedDPSTrainer(config=rllib_config, env=this_env, logger_creator=logger_creator)
        else:
            raise Exception('Unimplemented trainer')
    
    if 'checkpoint' in config and config['checkpoint'] is not None:
        print('restoring trainer')
        trainer.restore(config['checkpoint'])
        print('done')
    
    return environment, trainer
