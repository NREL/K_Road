import os

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
# controllers = ['stanley', 'pure_pursuit', 'pid', 'mpc_ltv', 'kino', 'mpc_lti', 'shivam_2018', 'nl_mpc']
controllers = ['stanley', 'pure_pursuit', 'pid', 'mpc_ltv', 'shivam_2018', 'nl_mpc']

for path_generator in path_generators:
    for controller in controllers:
        command = 'python run_classical_experiment.py {} {}'.format(path_generator, controller)
        print(command)
        os.system(command)
