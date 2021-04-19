import sys
# from plotter import ExperimentPlotter
from scenario.trajectory_tracking.experiment.plotter import ExperimentPlotter

sys.path.append('../../../')


"""
    This is how you load all data
    inside a folder

    In order for this to work properly, you need three files from each experiment:
     - *_layout.json
     - *_log.jsonl
     - *_config.json

     If any of these files are missing, then the plotter will ignore the experiment!
"""
# log_path = '/home/ctripp/project/cavs/src/cavs-environments/cavs_environments/vehicle/k_road/scenario/trajectory_tracking' \
#            '/experiment/pub_logs/basic/'
# output_path = '/home/ctripp/project/cavs/src/cavs-environments/cavs_environments/vehicle/k_road/scenario' \
#               '/trajectory_tracking/experiment/pub_logs/basic/fig'

log_path = '/home/ctripp/project/cavs/src/cavs-environments/cavs_environments/vehicle/k_road/scenario/trajectory_tracking/experiment/pub_logs/mpc/'
output_path = '/home/ctripp/project/cavs/src/cavs-environments/cavs_environments/vehicle/k_road/scenario/trajectory_tracking/experiment/pub_logs/mpc/fig/'

plotter = ExperimentPlotter(path_name=log_path, verbose=False)

"""
    This will generate all the path figures. It will sort the experiments by path and
        call `generate_speed_plots()`, `generate_rewards_plots()`, `generate_xy_plots()`,
        `generate_crosstrack_error()`, `generate_inputs_plots()`
"""

plotter.generate_all_path_figures(target_path=output_path)