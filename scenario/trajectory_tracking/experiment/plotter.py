import json_lines
import matplotlib
import pandas as pd
import os
import sys
import numpy as np
from scipy import stats
from typing import List
import json
import matplotlib.pyplot as plt

plt.ion()
plt.rcParams['figure.figsize'] = [10, 8]
plt.style.use('seaborn-whitegrid')
plt.ion()
plt.rcParams['figure.figsize'] = [10, 8]
font = {'family' : 'monospace',
        'weight' : 'medium',
        'size'   : 18}
matplotlib.rc('font', **font)
csfont = {'fontname':'Times New Roman', 'size' : 34}

class ExperimentPlotter:
    def __init__(self, path_name : str = 'bogus_results_file',
                            verbose : bool = True, labels_map : dict = {}):
        self.path_name = path_name
        self.verbose = verbose
        self.labels_map = labels_map
        self.test_dictionary = {}
        self.data = {}

    def generate_all_path_figures(self, target_path : str = 'figures', exclude_experiments : List[str] =[]):
        if not os.path.exists(target_path):
            os.mkdir(target_path)
        data = self.retrieve_all_info_under_file()
        self.data = self.append_experiment_statistics(data)
        self.test_dictionary = self.classify_runs_by_path(data)
        self.labels_map = self.generate_labels_map(data)
        self.generate_speed_plots(self.data, target_path, self.test_dictionary, exclude_experiments, self.labels_map)
        print("Generated speed plots!")
        self.generate_rewards_plots(self.data, target_path, self.test_dictionary, exclude_experiments, self.labels_map)
        print("Generated inputs plots!")
        self.generate_xy_plots(self.data, target_path, self.test_dictionary, exclude_experiments, self.labels_map)
        print("Generated XY plots!")
        self.generate_crosstrack_error(self.data, target_path, self.test_dictionary, exclude_experiments, self.labels_map)
        print("Generated crosstrack error plots!")
        self.generate_inputs_plots(self.data, target_path, self.test_dictionary, exclude_experiments, self.labels_map)
        print("Generated inputs plots!")

    def load_logfile_as_dataframe(self, file_path_name : str):
        rows = []
        with open(file_path_name, 'rb') as file:
            rows = [row for row in json_lines.reader(file)]

        columns = rows[0]
        data = rows[1:]

        df = pd.DataFrame(data, columns=columns)
        df.set_index('step_number', inplace=True)
        return df

    def unique(self, string_list : str):
        """
            INPUT: list of all relevant experiment files
            OUTPUT: Unique experiment names
        """
        unique_strings = {}
        for name in string_list:
            if '_config.json' in name:
                unique_substring = name.split('_config.json')
                unique_substring = unique_substring[0]
                if unique_substring not in unique_strings:
                    unique_strings[unique_substring] = True
        return list(unique_strings.keys())

    def parse_relevant_files(self):
        """
            INPUT: path to the experiment folder
            OUTPUT: relevant file names
        """
        return [f for f in os.listdir(self.path_name) if ('json' in f.split('.')[-1])]

    def organize_experiment_data(self, experiments_list : [str]):
        """
            INPUT: string with unique experiments done
            OUTPUT: json dictionary with all experiments organized
        """
        data = {}
        for exp in experiments_list:
            with open(os.path.join(self.path_name, exp + '_config.json'), encoding='utf-8') as fh:
                config_file = json.load(fh)
                data[exp] = {'metadata' : config_file}
                num_exp = config_file['num_runs']
            all_files = os.listdir(self.path_name)
            data[exp]['exp'] = {}
            for i in range(num_exp):
                data[exp]['exp'][str(i)] = {}
                data[exp]['exp'][str(i)]['data'] = self.load_logfile_as_dataframe(os.path.join(
                                self.path_name, exp + "_" + str(i) + "_log.jsonl"))
            if exp + "_" + str(i) + "_layout.json" in os.listdir(self.path_name):
                with open(os.path.join(self.path_name, exp + "_" + str(i) + "_layout.json")) as js:
                    data[exp]['exp'][str(i)]['road_data'] = json.load(js)
        return data

    def print_data_dict(self, data : dict, depth : int):
        sub_d = data
        tabs = ""
        if not isinstance(sub_d, dict):
            return
        else:
            if depth == 0:
                print(" ", next(iter(sub_d)), ":")
            depth += 1

            for key in sub_d.keys():
                self.print_data_dict(sub_d[key], depth)
                print(" "*(4*depth), key, ":")

    def retrieve_all_info_under_file(self):
        """
            INPUT  : str, bool = file path, verbose
            OUTPUT : dict      = aggregated data dictionary
        """
        folder_name = self.path_name
        unique_tests = self.parse_relevant_files()
        unique_tests = self.unique(unique_tests)
        data = self.organize_experiment_data(unique_tests)
        if self.verbose:
            print("_____________________")
            print("Displaying data tree:")
            print("_____________________")
            print()
            self.print_data_dict(data, 0)
        return data

    def isclosest(self, X, Y):
        """
            INPUT:  X --> points along path
                    Y --> minimum resolution path_point
            OUTPUT: Dataframe with True in the columns that will be selected and False otherwise
        """
        column = np.array([False]*X.shape[0])
        for y in Y:
            # This implements quicksort
            index = (X-y).abs().argsort()[:1]
            column[index] = True
        return column

    def aggregate_experiments(self, exp_subtree : dict, min_resolution : int,
                                    min_resolution_exp : str, column_names):
        """
            INPUT: Experiment sub-tree
            OUPTUT: Experiment sub-tree statistics as dataframes
            Info about statistics chosen here:
                https://www.princeton.edu/~cap/AEESP_Statchap_Peters.pdf
        """
        min_resolution_percentages = exp_subtree[min_resolution_exp]['data']['percentage_along_path']
        dataset = []
        for key in exp_subtree.keys():
            current_perc = exp_subtree[key]['data']['percentage_along_path']
            chosen_rows = self.isclosest(current_perc, min_resolution_percentages)

            new_dataframe = exp_subtree[key]['data'][chosen_rows]

            dataset += [new_dataframe.values.tolist()]

        dataset = np.array(dataset)
        if not all(dataset.shape):
            return {'mean' : [], 'standard_deviation' : [], 'standard_error_of_the_mean': []}
        else:
            return {'mean' : pd.DataFrame(data = np.mean(dataset, axis = 0), columns = column_names) ,
            'standard_deviation' : pd.DataFrame(data = np.std(dataset, axis = 0), columns = column_names) ,
                   'standard_error_of_the_mean' : pd.DataFrame(data = stats.sem(dataset, axis = 0), columns = column_names)}

    def append_experiment_statistics(self, data):
        """
            INPUT: dictionary of all raw experiment data
            OUTPUT: original dictionary + experiment statistics at minimum resolution
        """
        for key in data.keys():
            min_resolution = sys.maxsize #data[key]['exp'][list(data[key]['exp'].keys())[0]]['data'].shape[0]
            min_resolution_exp = '0'
            for kkey in data[key]['exp'].keys():
                current_resolution = data[key]['exp'][kkey]['data'].shape[0]

                if min_resolution > current_resolution:
                    min_resolution = current_resolution
                    min_resolution_exp = kkey

                delta_along_path = data[key]['exp'][kkey]['data']['delta_along_path']
                cum_delta = delta_along_path.cumsum(axis = 0)
                sum_delta = delta_along_path.sum(axis = 0)
                data[key]['exp'][kkey]['data']['accum_delta_along_path'] = cum_delta
                data[key]['exp'][kkey]['data']['percentage_along_path'] = cum_delta / sum_delta
                column_names = data[key]['exp'][kkey]['data'].columns

                data[key]['metadata']['min_resolution'] = min_resolution

                data[key]['summary'] = self.aggregate_experiments(data[key]['exp'], min_resolution,
                                                             min_resolution_exp, column_names)
                data[key]['summary']['columns'] = column_names.values

        return data

    def classify_runs_by_path(self, data): #, unique_path_generators: List[str] = None):
        # if unique_path_generators is None:
        #     unique_path_generators = ['curriculum_angled_path_factory', 'sine_path_generator', 'circle_path_factory',
        #                               'figure_eight_generator', 'carla_json_generator',
        #                               'straight_variable_speed_generator',
        #                               'left_lane_change_generator', 'right_lane_change_generator',
        #                               'snider_2009_track_generator',
        #                               'double_lane_change_generator', 'straight_variable_speed_pulse_generator',
        #                               'hairpin_turn_generator', 'right_turn_generator']
        # else:
        #     unique_path_generators = unique_path_generators

        test_dictionary = {}
        # for path in unique_path_generators:
        #     test_dictionary[path] = []

        # FIXME: Reduce complexity:
        for key in data.keys():
            path_generator = data[key]['metadata']['environment']['process']['path_generator']
            if path_generator not in test_dictionary:
                test_dictionary[path_generator] = []
            test_dictionary[path_generator].append(key)
        return test_dictionary

    def generate_labels_map(self, data):
        labels_map = {}
        for key in data.keys():
            labels_map[key] = []
            if key[0] == 'x':
                labels_map[key] += 'RL '
                if 'pure_pursuit' in key:
                    labels_map[key] += 'Pure Pursuit '
                if 'use_controller_action' in key:
                    labels_map[key] += []
                if 'w_' in key:
                    new_string = key.split('_w_')
                    new_string = new_string[1].split("_")
                    labels_map[key] += "N = " + new_string[0] + " "
                if 'steer' in key:
                    new_string = key.split("_steer_")
                    new_string = new_string[1].split("_accel")
                    steer_number = new_string[0].replace("_", ".")
                    labels_map[key] += "steer = " + steer_number + " "
                    new_string = new_string[1].split("_")
                    labels_map[key] += "accel = " + new_string[1] + "." + new_string[2] + " "
            else:
                if key.startswith('pure_pursuit'):
                    labels_map[key] += "Pure Pursuit "
                if key.startswith('stanley'):
                    labels_map[key] += "Stanley "
                if key.startswith('pid'):
                    labels_map[key] += "Proportional"
                if key.startswith('mpc_ltv'):
                    labels_map[key] += "MPC LTV"
                if key.startswith('mpc_lti'):
                    labels_map[key] += "MPC LTI"
                if key.startswith('shivam_2018'):
                    labels_map[key] += "Shivam 2018"

            labels_map[key] = ''.join(labels_map[key])
        return labels_map

    def generate_speed_plots(self, data, target_path, test_dictionary, exclude_experiments, labels_map):
        for path in test_dictionary.keys():
            #if len(test_dictionary[path]) > 0:
            plt.figure(figsize=(18, 16))
            for enum, key in enumerate(test_dictionary[path]):
                if key not in exclude_experiments:
                    if key not in labels_map:
                        label_this = data[key]['metadata']['controller']
                    else:
                        label_this = labels_map[key]

                    if len(data[key]['summary']['mean']) > 0:
                        target_speed = data[key]['summary']['mean']['process.target_speed']
                        mean_speed = data[key]['summary']['mean']['instantaneous_body_velocity[0]']

                        std_speed = data[key]['summary']['standard_deviation']['instantaneous_body_velocity[0]']
                        std_error = data[key]['summary']['standard_error_of_the_mean']['instantaneous_body_velocity[0]']
                        mean_percentage = data[key]['summary']['mean']['percentage_along_path']

                        if enum == len(test_dictionary[path]) - 1:
                            plt.plot(mean_percentage, target_speed, color='gray', linestyle='dashed', linewidth=4.,
                                     label=r'$\dot{x}_{reference}$')

                        plt.plot(mean_percentage, mean_speed, alpha=1, linewidth=4.,
                                 label=label_this)
                        plt.legend(fontsize=25)
                        plt.ylabel(r'Speed $[\frac{m}{s}]$', fontsize=10, **csfont)
                        if enum == 0:
                            plt.title('Instantaneous Speed vs. Reference Speed')

            plt.xlabel('Percentage along Path [%]', fontsize=10, **csfont)
            plt.xticks(fontsize=35)
            plt.yticks(fontsize=35)
            if not os.path.exists(os.path.join(target_path, 'speed')):
               os.mkdir(os.path.join(target_path, 'speed'))
            plt.savefig(os.path.join(target_path, 'speed', path + "_speed.svg"), format='svg')

    def generate_inputs_plots(self, data, target_path, test_dictionary, exclude_experiments, labels_map):
        for path in test_dictionary.keys():
            if len(test_dictionary[path]) > 0:
                plt.figure(figsize=(18, 16))
                for enum, key in enumerate(test_dictionary[path]):
                    if key not in exclude_experiments:
                        if key not in labels_map:
                            label_this = data[key]['metadata']['controller']
                        else:
                            label_this = labels_map[key]

                        if len(data[key]['summary']['mean']) > 0:
                            mean_acceleration = data[key]['summary']['mean']['action[0]'].apply(
                                lambda e: max(-1.0, min(1.0, e)))
                            mean_steering = data[key]['summary']['mean']['action[1]'].apply(
                                lambda e: max(-1.0, min(1.0, e)))
                            mean_percentage = data[key]['summary']['mean']['percentage_along_path']

                            plt.subplot(2, 1, 1)
                            plt.plot(mean_percentage, mean_acceleration, alpha=1, linewidth=2.5,
                                     label=label_this)

                            plt.legend(fontsize=24)
                            plt.ylabel(r'Norm. Accel.', fontsize=10, **csfont)

                            if enum == 0:
                                plt.title('Control Inputs vs. Percentage along Path', **csfont)

                            plt.subplot(2, 1, 2)
                            plt.plot(mean_percentage, mean_steering, alpha=1, linewidth=2.5,
                                     label=label_this)
                            plt.ylabel(r'Norm. Steer.', fontsize=10, **csfont)

                            plt.legend(fontsize=24)

                plt.xlabel('Percentage along Path [%]', **csfont)
                plt.xticks(fontsize=35)
                plt.yticks(fontsize=35)
                if not os.path.exists(os.path.join(target_path, 'inputs')):
                   os.mkdir(os.path.join(target_path, 'inputs'))
                plt.savefig(os.path.join(target_path, 'inputs', path  + "_inputs.svg"), format='svg')

    def generate_rewards_plots(self, data, target_path, test_dictionary, exclude_experiments, labels_map):
        for path in test_dictionary.keys():
            if len(test_dictionary[path]) > 0:
                plt.figure(figsize=(18, 16))
                for enum, key in enumerate(test_dictionary[path]):
                    if key not in exclude_experiments:
                        if key not in labels_map:
                            label_this = data[key]['metadata']['controller']
                        else:
                            label_this = labels_map[key]
                        if len(data[key]['summary']['mean']) > 0:
                            mean_reward = data[key]['summary']['mean']['reward']
                            std_reward = data[key]['summary']['standard_deviation']['reward']
                            std_error = data[key]['summary']['standard_error_of_the_mean']['reward']
                            mean_percentage = data[key]['summary']['mean']['percentage_along_path']

                            plt.fill_between(mean_percentage, -mean_reward - std_reward, -mean_reward + std_reward,
                                             alpha=1)
                            plt.plot(mean_percentage, -mean_reward, alpha=1., linewidth=3.5, \
                                     label=label_this)
                            plt.legend(fontsize=24)
                            plt.ylabel('cost', fontsize=10, **csfont)
                            if enum == 0:
                                plt.title('Performance vs. Average Track Percentage', **csfont)
                plt.xlabel('Percentage along Path [%]', **csfont)
                plt.xticks(fontsize=35)
                plt.yticks(fontsize=35)
                if not os.path.exists(os.path.join(target_path, 'reward')):
                   os.mkdir(os.path.join(target_path, 'reward'))
                plt.savefig(os.path.join(target_path, 'reward', path + '_reward.svg'), format='svg')

    def generate_xy_plots(self, data, target_path, test_dictionary, exclude_experiments, labels_map):
        for path in test_dictionary.keys():
            if len(test_dictionary[path]) > 0:
                plt.figure(figsize=(18, 16))
                for enum, key in enumerate(test_dictionary[path]):
                    if key not in exclude_experiments:
                        if key not in labels_map:
                            label_this = data[key]['metadata']['controller']
                        else:
                            label_this = labels_map[key]
                        if len(data[key]['summary']['mean'])>0:
                            mean_x = data[key]['summary']['mean']['position[0]']
                            mean_y = data[key]['summary']['mean']['position[1]']

                            std_x = data[key]['summary']['standard_deviation']['position[0]']
                            std_y = data[key]['summary']['standard_deviation']['position[1]']

                            std_error_x = data[key]['summary']['standard_error_of_the_mean']['position[0]']
                            std_error_y = data[key]['summary']['standard_error_of_the_mean']['position[1]']

                            plt.plot(mean_x, mean_y,alpha = 1., linewidth = 3.,label = label_this)

                            plt.legend(fontsize = 20, loc = 'upper right')
                            plt.ylabel('y [m]', **csfont)
                            if enum == 0:
                                plt.title('Road Travelled', **csfont)
                plt.xlabel('x [m]', **csfont)
                plt.xticks(fontsize = 35)
                plt.yticks(fontsize = 35)
                if not os.path.exists(os.path.join(target_path, 'position')):
                   os.mkdir(os.path.join(target_path, 'position'))
                plt.savefig(os.path.join(target_path, 'position',  path + '_xy.svg'), format = 'svg')

    def generate_crosstrack_error(self, data, target_path, test_dictionary, exclude_experiments, labels_map):
        for path in test_dictionary.keys():
            if len(test_dictionary[path]) > 0:
                plt.figure(figsize=(18, 16))
                for enum, key in enumerate(test_dictionary[path]):
                    if key not in exclude_experiments:
                        if key not in labels_map:
                            label_this = data[key]['metadata']['controller']
                        else:
                            label_this = labels_map[key]
                        if len(data[key]['summary']['mean']) > 0:
                            mean_crosstrack = data[key]['summary']['mean']['cross_track_error']
                            std_crosstrack = data[key]['summary']['standard_deviation']['cross_track_error']
                            std_crosstrack = data[key]['summary']['standard_error_of_the_mean']['cross_track_error']

                            mean_percentage = data[key]['summary']['mean']['percentage_along_path']

                            plt.fill_between(mean_percentage, mean_crosstrack - std_crosstrack,
                                             mean_crosstrack + std_crosstrack, \
                                             alpha=0.5)
                            plt.plot(mean_percentage, mean_crosstrack, \
                                     label=label_this, linewidth=2)
                            plt.legend(fontsize=18, loc='upper right')
                            plt.ylabel('Crosstrack Error', **csfont)

                            plt.title('Average Crosstrack Error [m]', **csfont)
                plt.xlabel('Percentage along Path [%]', **csfont)
                plt.xticks(fontsize=35)
                plt.yticks(fontsize=35)
                if not os.path.exists(os.path.join(target_path, 'crosstrack')):
                   os.mkdir(os.path.join(target_path, 'crosstrack'))
                plt.savefig(os.path.join(target_path, 'crosstrack', path + '_crosstrack_error.svg'), format='svg')

    def generate_tabular_entries(self, tabular_file_name : str = 'tabular',
                                 paths_to_tabulate : List[str] = [], test_dictionary : dict = {},
                                 exclude_experiments : List[str] = [], labels_map : dict = None):
        """
            ExperimentPlotter::generate_tabular_entries()

            This function outputs a LaTeX configured table with the summaries with the
            all the experiments in certain path.

            There is a RunTime error that appears while running this function, when this
            happens do not kill the code; wait for the code to finish.

        """
        data = self.retrieve_all_info_under_file()
        self.data = self.append_experiment_statistics(data)
        self.test_dictionary = self.classify_runs_by_path(data)
        self.labels_map = self.generate_labels_map(data)
        tabular_file = open(tabular_file_name + ".tex", "w")
        num_paths = len(paths_to_tabulate)

        header_string = [r"\begin{tabular}{|l"] + ["|c"]*num_paths + ["|l|}\hline\n"]
        print("Values from plot recorder: ")
        print(data[list(data.keys())[0]]['exp']['0']['data'].columns.values)
        second_header_string = ["&"]
        third_header_string = ["&"]
        header_string = ''.join(header_string)
        for enum, path in enumerate(paths_to_tabulate):
            second_header_string += [r"\multicolumn{10}{c|}{" +  path + "}"]
            third_header_string += [r"\bar{\dot{\delta}} & \bar{\dot{a}} &  \bar{\epsilon}_v & " +
                                    r"\epsilon_{v, max} & \epsilon_{v, min} & \bar{cs} & cs_{max} &" +
                                    r"cs_{min} & \epsilon_{\Psi, max} " +
                                    "& \epsilon_{\Psi, min}"]
            if enum < len(paths_to_tabulate) - 1:
                second_header_string += ["&"]
                third_header_string += ["&"]
            else:
                second_header_string += [r"\\"+ "\n"]
                third_header_string += [r"\\" +  "\n" + r"\hline" + "\n"]

            row = []
            for numm, experiment in enumerate(self.test_dictionary[path]):
                print(len(self.data[experiment]['summary']['mean']))
                if len(self.data[experiment]['summary']['mean']) > 0:
                    # SARAH : Values for latex tables
                    mean_crosstrack = np.nanmean(self.data[experiment]['summary']['mean']['cross_track_error'])
                    max_crosstrack = self.data[experiment]['summary']['mean']['cross_track_error'].max()
                    min_crosstrack = self.data[experiment]['summary']['mean']['cross_track_error'].min()
                    elapsed = self.data[experiment]['summary']['mean']['elapsed']

                    # FIXME: Divide by timestep length
                    mean_del_steer = np.nanmean(np.diff(self.data[experiment]['summary']['mean']['action[1]'], n=1))
                    mean_del_acc = np.nanmean(np.diff(self.data[experiment]['summary']['mean']['action[0]'], n=1))

                    mean_speed_error = np.nanmean(self.data[experiment]['summary']['mean']['speed_error'])
                    min_speed_error = self.data[experiment]['summary']['mean']['speed_error'].min()
                    max_speed_error = self.data[experiment]['summary']['mean']['speed_error'].max()

                    max_yaw_error = (self.data[experiment]['summary']['mean']['yaw'] - \
                                     self.data[experiment]['summary']['mean']['cross_track_yaw_heading']).max()
                    min_yaw_error = (self.data[experiment]['summary']['mean']['yaw'] - \
                                     self.data[experiment]['summary']['mean']['cross_track_yaw_heading']).min()
                    mean_yaw_error = np.nanmean(self.data[experiment]['summary']['mean']['yaw'] - \
                                     self.data[experiment]['summary']['mean']['cross_track_yaw_heading'])

                    if experiment in self.labels_map.keys():
                        row += [self.labels_map[experiment]]
                    else:
                        row += [experiment]
                    row += ["&"]
                    row += ["{:.2e}".format(mean_del_steer) + " & " + "{:.2e}".format(mean_del_acc) +  " & " + \
                           "{:.2e}".format(mean_speed_error) + " & " + "{:.2e}".format(max_speed_error) +  " & " +\
                           "{:.2e}".format(min_speed_error) + " & " + "{:.2e}".format(mean_crosstrack) + " & " +\
                           "{:.2e}".format(max_crosstrack) + " & " + "{:.2e}".format(min_crosstrack) + " & " +\
                           "{:.2e}".format(max_yaw_error) + " & " + "{:.2e}".format(min_yaw_error) ]

                    row += [r"\\" + "\n"]

        row = ''.join(row)
        last_lines = r"\hline \end{tabular}"
        second_header_string = ''.join(second_header_string)
        third_header_string = ''.join(third_header_string)
        tabular_file.write(header_string)
        tabular_file.write(second_header_string)
        tabular_file.write(third_header_string)
        tabular_file.write(row)
        tabular_file.write(last_lines)
        print("Generated the LaTeX formatted table!")

    def classify_runs_by_waypoint_number(self, data):
        test_dictionary = {}
        minimum_n_mpc = sys.maxsize
        minimum_n_mpc_exp = ""
        minimum_n_rl = sys.maxsize
        minimum_n_rl_exp = ""
        for key in data.keys():
            if key.startswith("mpc_ltv"):
                if "OSQP_MPC_LTV" not in test_dictionary.keys():
                    test_dictionary["OSQP_MPC_LTV"] = {}
                split_string = key.split("_")
                index = split_string.index("n")
                n = int(split_string[index+1])
                index = split_string.index("nc")
                nc = int(split_string[index + 1])
                if n < minimum_n_mpc:
                    minimum_n_mpc = n
                    minimum_n_mpc_exp = key
                test_dictionary["OSQP_MPC_LTV"][(n, nc)] = key
            if key.startswith("x"):
                if "RL" not in test_dictionary.keys():
                    test_dictionary["RL"] = {}
                split_string = key.split("_")
                index = split_string.index("w")
                n = int(split_string[index + 1])
                if n < minimum_n_rl:
                    minimum_n_rl = n
                    minimum_n_rl_exp = key
                test_dictionary["RL"][n] = key
        if minimum_n_rl < sys.maxsize:
            test_dictionary["RL"]['min_n'] = {}
            test_dictionary["RL"]['min_n']['n'] = minimum_n_rl
            test_dictionary["RL"]['min_n']['exp'] = minimum_n_rl_exp
        if minimum_n_mpc < sys.maxsize:
            test_dictionary["OSQP_MPC_LTV"]['min_n'] = {}
            test_dictionary["OSQP_MPC_LTV"]['min_n']['n'] = minimum_n_mpc
            test_dictionary["OSQP_MPC_LTV"]['min_n']['exp'] = minimum_n_mpc_exp
        return test_dictionary

    def generate_cpu_comparison_plots(self, target_path = 'cpu_time_comparison'):
        data = self.retrieve_all_info_under_file()
        self.data = self.append_experiment_statistics(data)
        self.test_dictionary = self.classify_runs_by_waypoint_number(data)

        plt.figure(figsize=(18, 16))
        min_experiment_prefix_mpc = self.test_dictionary["OSQP_MPC_LTV"]['min_n']['exp']
        min_experiment_prefix_rl = self.test_dictionary["RL"]['min_n']['exp']

        min_elapsed_mpc = self.data[min_experiment_prefix_mpc]['summary']['mean']['elapsed'].mean()
        min_reward_mpc = self.data[min_experiment_prefix_mpc]['summary']['mean']['reward'].sum()
        min_elapsed_rl = self.data[min_experiment_prefix_rl]['summary']['mean']['elapsed'].mean()
        min_reward_rl = self.data[min_experiment_prefix_rl]['summary']['mean']['reward'].sum()

        for key in self.test_dictionary["OSQP_MPC_LTV"].keys():
            if isinstance(key, tuple):
                if key[1] == 1:
                    elapsed = self.data[self.test_dictionary["OSQP_MPC_LTV"][key]]['summary']['mean']['elapsed'].mean()/min_elapsed_mpc
                    reward = self.data[self.test_dictionary["OSQP_MPC_LTV"][key]]['summary']['mean']['reward'].sum()
                    plt.scatter(reward, elapsed, marker = 'x', linewidth = 30., label = "OSQP MPC LTV Nc = N="+str(key[0]))
        plt.ylabel(r"$\frac{Mean Ctrl. Loop CPU Time [s]}{Mean Ctrl. Loop CPU Time at N =" + \
                    str(self.test_dictionary["OSQP_MPC_LTV"]['min_n']['n']) +  "[s]}$", fontsize=10, **csfont)
        plt.title('Scaling of Mean CPU Ctrl. Loop Time vs. Reward', **csfont)
        plt.xlabel('Total Reward', **csfont)
        plt.legend(fontsize=24)
        plt.xticks(fontsize=35)
        plt.yticks(fontsize=35)
        plt.xlim([-600., 120.])
        plt.savefig(target_path + '_1.svg', format='svg')
        plt.figure(figsize=(18, 16))
        for key in self.test_dictionary["OSQP_MPC_LTV"].keys():
            if isinstance(key, tuple):
                if key[1] > 1 or key[0] == 1:
                    elapsed = self.data[self.test_dictionary["OSQP_MPC_LTV"][key]]['summary']['mean']['elapsed'].mean()/min_elapsed_mpc
                    reward = self.data[self.test_dictionary["OSQP_MPC_LTV"][key]]['summary']['mean']['reward'].sum()
                    plt.scatter(reward, elapsed,marker = 'x', linewidth = 30.,label="OSQP MPC LTV Nc = 1, N=" + str(key[0]))
        plt.ylabel(r"$\frac{Mean Ctrl. Loop CPU Time [s]}{Mean Ctrl. Loop CPU Time at N =" + \
                   str(self.test_dictionary["OSQP_MPC_LTV"]['min_n']['n']) + "[s]}$", fontsize=10, **csfont)
        plt.title('Scaling Mean CPU Ctrl. Loop Time vs. Reward', **csfont)
        plt.xlabel('Total Reward', **csfont)
        plt.legend(fontsize=24,loc='upper left')
        plt.xticks(fontsize=35)
        plt.yticks(fontsize=35)
        plt.xlim([-600., 120.])
        plt.savefig(target_path + '_2.svg', format='svg')
        #for key in self.test_dictionary["RL"].keys():
        #    elapsed = self.data[self.test_dictionary["RL"][key]]['summary']['mean']['elapsed'].mean()/min_elapsed_rl
        #    reward = self.data[self.test_dictionary["RL"][key]]['summary']['mean']['elapsed'].mean() / min_reward_rl
        #    plt.plot(reward, elapsed, marker = 'x', linewidth = 30., label = "RL N = " + str(key))
        plt.ylabel('Mean Ctrl. Loop CPU Time [s]', fontsize=10, **csfont)
        plt.title('Mean CPU Ctrl. Loop Time vs. Reward', **csfont)
        plt.xlabel('Total Reward', **csfont)
        plt.legend(fontsize=24, loc='upper right')
        plt.xticks(fontsize=35)
        plt.yticks(fontsize=35)
        plt.xlim([-600., 120.])
        #plt.savefig(target_path + '.svg', format='svg')


if __name__ == '__main__':
    plotter = ExperimentPlotter(path_name='aug14_mpc_cpu_plot_experiments', verbose=False)
    plotter.generate_all_path_figures(target_path='august_14_plots')
    plotter.generate_tabular_entries('tabular',
                                     ['double_lane_change_generator'],
                                     plotter.test_dictionary, [],
                                     plotter.labels_map)
    plotter.generate_cpu_comparison_plots(target_path = 'cpu_time_comparison')
