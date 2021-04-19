import json
import os
import sys
import time
from copy import deepcopy
from pprint import pprint
# import matplotlib.pyplot as plt
from typing import Optional

from command_line_tools import command_line_config
from data_logging.JSON_lines_record_logger import JSONLinesRecordLogger
from data_logging.data_recorder import DataRecorder

log_path = os.path.join(os.path.curdir, 'log')
if not os.path.exists(log_path):
    # noinspection PyBroadException
    try:
        os.makedirs(log_path)
    except:
        print('Error creating log dir "', log_path, '". This might be okay if you ran multiple runners simultaneously.')


# def setup_sub_run(default_config: {}) -> ({}, DataRecorder):
#     '''
#     Sets up a config, logging prefix, and recorder for this run.
#     :param default_config:
#     :return:
#     '''
#     config = command_line_config.parse_config_from_args(sys.argv[1:], default_config)
#     pprint(config)
#     run_prefix = get_run_prefix(config, None)
#     write_config_log(config, run_prefix)
#     recorder = make_data_recorder(run_prefix)
#     return config, run_prefix, recorder

def setup_run(default_config: {}, subrun_name: Optional[str] = None, use_command_line: bool = True):
    '''
    Sets up a config and logging prefix for this run. Writes the config to a log file.
    :param default_config:
    :param subrun_name:
    :return:
    '''
    if use_command_line:
        config = command_line_config.parse_config_from_args(sys.argv[1:], default_config)
    else:
        config = deepcopy(default_config)
    
    pprint(config)
    run_prefix = get_run_prefix(config, subrun_name)
    write_config_log(config, run_prefix)
    return config, run_prefix


def make_data_recorder(run_prefix: str) -> DataRecorder:
    '''
    Makes a DataRecorder for logging this run
    :param run_prefix: the run log file prefix
    :return: a DataRecorder for this run
    '''
    log_filename = os.path.join(log_path, run_prefix + 'log' + '.jsonl')
    return DataRecorder(JSONLinesRecordLogger(log_filename))


def write_config_log(config: {}, run_prefix: str, suffix: str = 'config') -> None:
    '''
    Writes a json log file containing the configuration of this run
    :param config: run config
    :param run_prefix: run prefix
    '''
    config_filename = os.path.join(log_path, run_prefix + suffix + '.json')
    with open(config_filename, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2, sort_keys=True)


def get_run_prefix(config: {}, subrun_name: Optional[str] = None) -> str:
    '''
    Makes the run log file prefix string by concatenating the run name, subrun name (if it exists), and the current
    time
    :param config: run config
    :param run_prefix: run prefix
    :return: run log file prefix
    '''
    start_time = int(time.time() * 10000)
    subrun_name = '' if subrun_name is None else '_' + subrun_name
    return config['name'] + subrun_name + '_' + str(start_time) + '_'

# def plot_from_logger(recorder, x_axis, y_axis):
#     plt.figure()
#     plt.plot(recorder.get_column(x_axis), recorder.get_column(y_axis))
#     print(x_axis, recorder.get_column(x_axis))
#     print(y_axis, recorder.get_column(y_axis))
#     plt.xlabel(x_axis, fontsize=15)
#     plt.ylabel(y_axis, fontsize=15)
