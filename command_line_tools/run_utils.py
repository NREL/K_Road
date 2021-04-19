import errno
import json
import os
import sys
from datetime import datetime
from typing import Optional

from command_line_tools.command_line_config import parse_config_from_args


def makedir_if_not_exists(filename: str) -> None:
    try:
        os.makedirs(filename)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_run(
        default_config: {},
        output_path: Optional[str] = None,
        run_suffix: Optional[str] = None,
        place_in_subdir: bool = True,
        output_dir_name: str = 'log',
        ) -> ({}, str):
    '''
    Sets up a config and logging prefix for this run. Writes the config to a log file.
    :param default_config: the base configuration that the command line params override
    :param output_path: path to place output files including logs and configuration record. If None, './output' is used.
    :param run_suffix: appended to the run name. If None, then the ISO 8601 datetime is used with '.' instead of ':'.
    :param place_in_subdir: True to output into a subdir of output_path, False to directly into output_path
    :return: config, output_path, run_name
    '''
    
    config = parse_config_from_args(sys.argv[1:], default_config)
    run_name = config['name']
    run_suffix = '_' + datetime.now().isoformat().replace(':', '.') if run_suffix is None else run_suffix
    run_name += run_suffix
    
    output_path = os.path.join(os.path.curdir, output_dir_name) if output_path is None else output_path
    output_path = output_path if place_in_subdir and run_name is None else os.path.join(output_path, run_name)
    makedir_if_not_exists(output_path)
    
    print('setup_run() run_name: "' + run_name + '"')
    print('setup_run() output_path: "' + output_path + '"')
    # print('setup_run() config:')
    # pprint(config)
    
    write_config_log(config, output_path, run_name)
    print('setup_run() complete.')
    return config, output_path, run_name


def write_config_log(
        config: {},
        output_path: str,
        run_name: str,
        config_name: str = '_config') -> None:
    '''
    Writes a json log file containing the configuration of this run
    :param output_path: where to write the file
    :param run_name: prefix of the filename
    :param config_name: suffix of the filename
    '''
    config_filename = os.path.join(output_path, run_name + config_name + '.json')
    with open(config_filename, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2, sort_keys=True)