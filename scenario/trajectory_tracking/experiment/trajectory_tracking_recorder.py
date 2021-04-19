import json
import os
import time
from pprint import pprint

import ray

from k_road.entity.vehicle.vehicle_dbm3 import VehicleDBM3

from k_road.util import *
from command_line_tools.command_line_config import merge_configs
from command_line_tools.run_tools import (
    make_data_recorder,
    setup_run,
    write_config_log,
)
from scenario.trajectory_tracking.experiment.experiment_common import list_of_vectors_to_list_of_tuples, \
    make_environment_and_controller, common_default_config
from scenario.trajectory_tracking.trajectory_tracking_process import TrajectoryTrackingProcess


def run(config, run_prefix, run_number):
    # config['environment']['max_cross_track_error'] = 20.0
    # config['environment']['max_speed_error'] = 100.0

    config['environment']['max_cross_track_error'] = 20000000.0
    config['environment']['max_speed_error'] = 100000.0

    print('run() config:')
    pprint(config)

    subrun_prefix = run_prefix + str(run_number) + '_'

    recorder = make_data_recorder(subrun_prefix)
    recorder.add_columns(
        'step_number',
        'is_terminal',
        'reward',
        'raw_action[0]',
        'raw_action[1]',
        'action[0]',
        'action[1]',
        'position[0]',
        'position[1]',
        'yaw',
        'mean_velocity[0]',
        'mean_velocity[1]',
        'mean_angular_velocity',
        'instantaneous_body_velocity[0]',
        'instantaneous_body_velocity[1]',
        'instantaneous_global_velocity[0]',
        'instantaneous_global_velocity[1]',
        'instantaneous_angular_velocity',
        'mean_velocity_heading',
        'instantaneous_velocity_heading',
        'delta_along_path',
        'cross_track_front_error',
        'cross_track_error',
        'cross_track_rear_error',
        'process.target_speed',
        'speed_error',
        'penalty',
        'cross_track_yaw_heading',
        'cross_track_instantaneous_velocity_heading',
        'cross_track_mean_velocity_heading',
        'path_angle',
        'path_yaw_heading',
        'path_mean_velocity_heading',
        'path_instantaneous_velocity_heading',
        'accumulated_waypoint_number',
        'elapsed',
        'internal_speed'
    )
    ray.services.get_node_ip_address = lambda: '127.0.0.1'

    # if not ray.is_initialized():
    #   ray.init(ignore_reinit_error=True)

    recorder.set_schema()

    environment, trainer = make_environment_and_controller(config, None)
    # environment.components.append(controller)  # causes the controller to be rendered() properly

    render = config['render']
    max_run_length = config['max_run_length']

    process = environment.process
    observation = environment.reset()
    if process.road is not None:
        write_config_log({
            'segment_points': list_of_vectors_to_list_of_tuples(process.road.segment_points),
        }, subrun_prefix, suffix='layout')

    if render:
        environment.render()
    previous_cross_track_position = process.ego_vehicle.position

    is_terminal = False
    step_number = 0
    reward = 0
    accumulated_waypoint_number = 0

    # d_acceleration = []
    # d_steering = []
    #
    # d_x = []
    # d_y = []
    #
    # d_vx = []
    # d_vy = []
    # d_angular_velocity = []
    # d_yaw = []

    print('running...')
    while not is_terminal and step_number < max_run_length:
        start = time.perf_counter()
        raw_action = trainer.compute_action(observation)
        action = environment.compute_transformed_action(raw_action)
        elapsed = time.perf_counter() - start
        # print("elapsed: ", elapsed)
        process: TrajectoryTrackingProcess = environment.process
        ego_vehicle: VehicleDBM3 = process.ego_vehicle

        position = ego_vehicle.position
        yaw = ego_vehicle.angle

        mean_velocity = ego_vehicle.velocity
        mean_angular_velocity = ego_vehicle.angular_velocity

        instantaneous_body_velocity = ego_vehicle.instantaneous_body_velocity
        instantaneous_angular_velocity = ego_vehicle.instantaneous_angular_velocity
        instantaneous_global_velocity = ego_vehicle.instantaneous_global_velocity

        mean_velocity_heading = mean_velocity.angle
        instantaneous_velocity_heading = instantaneous_global_velocity.angle
        cross_track = process.cross_track
        cross_track_position = cross_track.point

        # count the waypoints that have been swept
        if previous_cross_track_position != cross_track_position:
            accumulated_waypoint_number += 1
        internal_speed = ego_vehicle.internal_longitudinal_velocity
        delta_along_path = (cross_track_position - previous_cross_track_position).length
        speed_error = process.target_speed - ego_vehicle.internal_longitudinal_velocity

        # cross_track_front_error = (process.cross_tracks[0].point - ego_vehicle.positions[0]).length

        cross_track_delta = process.cross_track.point - position
        cross_track_error = (process.cross_track.point - ego_vehicle.positions[0]).length

        # cross_track_rear_error = (process.cross_tracks[2].point - ego_vehicle.positions[2]).length

        penalty = cross_track_error * environment.rewarder.speed_tradeoff + fabs(speed_error)

        cross_track_yaw_heading = signed_delta_angle(cross_track_delta.angle, yaw)
        cross_track_instantaneous_velocity_heading = signed_delta_angle(cross_track_delta.angle,
                                                                        instantaneous_velocity_heading)
        cross_track_mean_velocity_heading = signed_delta_angle(cross_track_delta.angle, mean_velocity_heading)

        path_segment = cross_track.shape.body.entity
        path_angle = path_segment.direction.angle
        path_yaw_heading = signed_delta_angle(path_angle, yaw)
        path_mean_velocity_heading = signed_delta_angle(path_angle, mean_velocity_heading)
        path_instantaneous_velocity_heading = signed_delta_angle(path_angle, instantaneous_velocity_heading)

        observation, reward, is_terminal, extra = environment.step_with_transformed_action(action)

        # d_acceleration.append(action[0])
        # d_steering.append(action[1])
        # d_x.append(ego_vehicle.position[0])
        # d_y.append(ego_vehicle.position[1])
        # d_vx.append(ego_vehicle.internal_longitudinal_velocity)
        # d_vy.append(ego_vehicle.internal_lateral_velocity)
        # d_angular_velocity.append(ego_vehicle.angular_velocity)
        # d_yaw.append(ego_vehicle.angle)

        recorder.accumulate(
            step_number,
            is_terminal,
            reward,
            raw_action[0],
            raw_action[1],
            environment.action[0],
            environment.action[1],
            position[0],
            position[1],
            yaw,
            mean_velocity[0],
            mean_velocity[1],
            mean_angular_velocity,
            instantaneous_body_velocity[0],
            instantaneous_body_velocity[1],
            instantaneous_global_velocity[0],
            instantaneous_global_velocity[1],
            instantaneous_angular_velocity,
            mean_velocity_heading,
            instantaneous_velocity_heading,
            delta_along_path,
            cross_track_error,
            cross_track_error,
            cross_track_error,
            process.target_speed,
            speed_error,
            penalty,
            cross_track_yaw_heading,
            cross_track_instantaneous_velocity_heading,
            cross_track_mean_velocity_heading,
            path_angle,
            path_yaw_heading,
            path_mean_velocity_heading,
            path_instantaneous_velocity_heading,
            accumulated_waypoint_number,
            elapsed,
            internal_speed
        )

        print('{}'.format(mean_velocity))
        recorder.store()
        # previous_position = position
        previous_cross_track_position = cross_track_position

        if render:
            environment.render()
        # controller.render(environment.process, environment.view)

        step_number += 1
    print('finished after {} steps.'.format(step_number))

    # pygame.time.wait(1000 * 100)
    recorder.accumulate(step_number, is_terminal, reward, 0, 0)
    recorder.store()
    environment.close()

    # pyplot.rcParams['figure.figsize'] = 10, 10
    # # pyplot.figure(figsize=(20, 10))
    #
    # fig, axs = pyplot.subplots(7)
    # p = 0
    #
    # axs[p].plot(d_acceleration)
    # axs[p].set(ylabel='accel')
    # p += 1
    #
    # axs[p].plot(d_steering)
    # axs[p].set(ylabel='steer')
    # p += 1
    #
    # axs[p].plot(d_vx)
    # axs[p].set(ylabel='vx')
    # p += 1
    #
    # axs[p].plot(d_vy)
    # axs[p].set(ylabel='vy')
    # p += 1
    #
    # axs[p].plot(d_angular_velocity)
    # axs[p].set(ylabel='omega')
    # p += 1
    #
    # axs[p].plot(d_x)
    # axs[p].set(ylabel='x')
    # p += 1
    #
    # axs[p].plot(d_y)
    # axs[p].set(ylabel='y')
    # p += 1
    #
    # pyplot.show()
    #
    # pyplot.plot(d_x, d_y)
    # pyplot.title('position')
    # pyplot.xlim(-300, 600)
    # pyplot.ylim(-300, 200)
    # pyplot.show()


def parse_json_rl_controllers(file_path):
    import os

    rl_controllers = {}
    for file in os.listdir(file_path):
        if file != '.DS_Store':
            full_file_path = os.path.join(file_path, file)
            rl_controllers[file] = parse_rl_controller_checkpoint(full_file_path)
    return rl_controllers


def parse_rl_controller_checkpoint(file):
    info = {}
    info['file'] = file
    for subfile in os.listdir(file):
        if "_config.json" in subfile:
            config_f = open(os.path.join(file, subfile), )
            info['config'] = json.load(config_f)
        if "checkpoint" in subfile:
            other_subfile = subfile.split("_")
            new_subfile = other_subfile[0] + "-" + other_subfile[1]
            info['checkpoint'] = os.path.join(file, subfile, new_subfile)
            print('parse_rl_controller_checkpoint\n{}\n{}\n{}\n{}\n'.format(
                file, subfile, new_subfile, info['checkpoint']))
    return info


def run_checkpoint_experiments(rl_controller_dictionary={}, path_generators=['figure_eight_generator'], render=False,
                               default_config={}):
    if path_generators is None:
        paths = ['curriculum_angled_path_factory', 'sine_path_generator', 'circle_path_factory',
                 'figure_eight_generator', 'carla_json_generator', 'straight_variable_speed_generator',
                 'left_lane_change_generator', 'right_lane_change_generator', 'snider_2009_track_generator',
                 'double_lane_change_generator', 'straight_variable_speed_pulse_generator',
                 'hairpin_turn_generator', 'right_turn_generator']
    else:
        paths = path_generators
    for info in rl_controller_dictionary.values():
        for path_generator in paths:
            run_checkpoint_experiment(default_config, info, path_generator, render)


def run_checkpoint_experiment(default_config, info, path_generator, render):
    config = info['config']
    original_name = config['name']
    name = original_name + "_" + path_generator
    config['name'] = name
    config['checkpoint'] = info['checkpoint']
    config['environment']['process']['path_generator'] = path_generator
    config['render'] = render
    config['environment']['rewarder']['steering_control_effort_tradeoff'] = 0.0
    config['environment']['rewarder']['acceleration_control_effort_tradeoff'] = 0.0
    default_config = merge_configs(default_config, config)
    config, run_prefix = setup_run(default_config, use_command_line=False)
    try:
        print('Running experiment: {} | {}'.format(name, config['name']))
        evaluations = [run(config, run_prefix, i) for i in range(config['num_runs'])]
        print("Experiment " + name + " finished!")
    except Exception as e:
        print("Experiment " + name + " didn't run, exception '{}'".format(repr(e)))
    info['config']['name'] = original_name


def run_classical_experiments(
        path_generators=None,
        render=False,
        default_config={},
        controllers=None,
):
    if path_generators is None:
        path_generators = ['curriculum_angled_path_factory', 'sine_path_generator', 'circle_path_factory',
                           'figure_eight_generator', 'carla_json_generator', 'straight_variable_speed_generator',
                           'left_lane_change_generator', 'right_lane_change_generator', 'snider_2009_track_generator',
                           'double_lane_change_generator', 'straight_variable_speed_pulse_generator',
                           'hairpin_turn_generator', 'right_turn_generator']
    if controllers is None:
        controllers = ['stanley', 'pure_pursuit', 'pid', 'mpc_ltv']

    for control in controllers:
        for path in path_generators:
            default_config = merge_configs(default_config, {
                'num_runs': 1,
                'render': render,
                'controller': control,
                'name': control + "_" + path,
                'environment': {
                    'max_distance_error': 5.0,
                    'process': {
                        'path_generator': 'figure_eight_generator',
                    }
                }
            })

            config, run_prefix = setup_run(default_config)
            evaluations = [run(config, run_prefix, i) for i in range(config['num_runs'])]
    print("All classical controllers ran!")


######################################################################
#     SARAH: These are to run the experiments in a single processor
#               without manually specifying the config
######################################################################

# paths = ['curriculum_angled_path_factory', 'sine_path_generator', 'circle_path_factory',
#                 'figure_eight_generator', 'carla_json_generator', 'straight_variable_speed_generator',
#                 'left_lane_change_generator', 'right_lane_change_generator', 'snider_2009_track_generator',
#                 'double_lane_change_generator', 'straight_variable_speed_pulse_generator',
#                 'hairpin_turn_generator', 'right_turn_generator']

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
#
# rl_config_dictionary = parse_json_rl_controllers("sept29_2020_controllers")
# # rl_config_dictionary = parse_json_rl_controllers("oct2_2020_controllers")
# run_checkpoint_experiments(rl_controller_dictionary=rl_config_dictionary,
#                    path_generators=path_generators,
#                    render=False,
#                    common_default_config=common_default_config)
# run_classical_experiments(common_default_config=common_default_config, path_generators=path_generators)
######################################################################

# default_config = merge_configs(common_default_config, {
#     'num_runs':    1,
#     'render':      True,
#     'controller':  'mpc_ltv',
#     'name':        'mpc_ltv_n_2_nc_1',
#     'environment': {
#         'terminator' : {},
#         'controller' : {
#             'Q' : [[100., 0., 0.],
#                    [0., 100., 0.],
#                    [0., 0., 4000.]],
#             'R' : [[1., 0.],
#                    [0., 5e-4]],
#             'R_bar' : [[1., 0.],
#                        [0., 1e-1]],
#             'umin' : [-1.5, -np.deg2rad(37.)],
#             'umax' : [1., np.deg2rad(37.)],
#             'delumin' : [-3., -np.deg2rad(10.)],
#             'delumax' : [1.5, np.deg2rad(10.)],
#             'N' : 2,
#             'Nc' : 1,
#             'output_states' : [0, 1, 1, 0, 1, 0],
#         },
#         'max_distance_error': 25.0,
#         'process':            {
#             'path_generator': 'double_lane_change_generator',
#             }
#         }
#     })

# default_config = merge_configs(common_default_config, {
#     'num_runs':    1,
#     'render':      True,
#     'controller':  'mpc_ltv',
#     'name':        'mpc_ltv_n_2_nc_1',
#     'environment': {
#         'terminator':         {},
#         'controller':         {
#             },
#         'max_distance_error': 25.0,
#         'process':            {
#             'path_generator': 'double_lane_change_generator',
#             }
#         }
#     })

# default_config = merge_configs(common_default_config, {
#     'num_runs': 1,
#     'render':   True,
#     })
if __name__ == "__main__":
    default_config = merge_configs(common_default_config, {
        'num_runs': 1,
        'render': False,
        'controller': 'scheduled',
        'name': 'scheduled',
        'environment': {
            'terminator': {},
            'controller': {
                # 'input_csv': '/home/ctripp/project/cavs/src/cavs-environments/cavs_environments/vehicle/k_road/controller'
                #             '/scheduled_controller_inputs.csv'
            },
            'max_distance_error': 50000000000000000.0,
            'process': {
                'path_generator': 'straight_variable_speed_pulse_generator',
            }
        }
    })

    config, run_prefix = setup_run(default_config)
    evaluations = [run(config, run_prefix, i) for i in range(config['num_runs'])]

    ##To run multuple runs in parallel
    # pool = multiprocessing.Pool(
    #    processes=multiprocessing.cpu_count() if config['num_cpus'] is None else config['num_cpus'])
    #
    # with pool:
    #    evaluations = pool.map(functools.partial(run, config, run_prefix), [i for i in range(config['num_runs'])])


    if ray.is_initialized():
        ray.shutdown()
