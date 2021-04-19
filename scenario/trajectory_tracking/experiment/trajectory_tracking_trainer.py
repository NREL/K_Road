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

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from command_line_tools.run_tools import setup_run
from scenario.trajectory_tracking.experiment.experiment_common import setup_environment
from trainer.coordinated_dps_trainer import CoordinatedDPSTrainer
from trainer.es_actual import ESActualTrainer
from trainer.es_co_trainer import ESCOTrainer

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# ray.init(num_gpus=1)


# def make_trainer(rllib_config) -> Trainer:
#     config = rllib_config['env_config']['config']  # unwrap config from rllib config
#     config, this_env = setup_environment(config)
#
#     results_path = DEFAULT_RESULTS_DIR
#     checkpoint_path = results_path
#     # checkpoint_path = os.path.join(results_path, config['group'])
#     # checkpoint_path = os.path.join(checkpoint_path, config['name'])
#
#     logdir_prefix = config['group'] + '_' + config['name'] + '_' + datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
#     #
#     print('checkpoint_path:', checkpoint_path, 'results_path', results_path, 'logdir_prefix', logdir_prefix)
#
#     def logger_creator(config):
#         if not os.path.exists(checkpoint_path):
#             os.makedirs(checkpoint_path)
#         logdir = tempfile.mkdtemp(
#             prefix=logdir_prefix, dir=checkpoint_path)
#         print('logger creator: ', logdir)
#         return UnifiedLogger(config, logdir, None)
#
#     controller_setting = config['controller']
#     if controller_setting == 'ESActualTrainer':
#         trainer = ESActualTrainer(config=rllib_config, env=this_env)
#     elif controller_setting == 'ESCOTrainer':
#         trainer = ESCOTrainer(config=rllib_config, env=this_env, logger_creator=logger_creator)
#     elif controller_setting == 'CoordinatedDPSTrainer':
#         trainer = CoordinatedDPSTrainer(config=rllib_config, env=this_env, logger_creator=logger_creator)
#     else:
#         raise Exception('Unimplemented controller')
#
#     if 'checkpoint' in config and config['checkpoint'] is not None:
#         print('restore trainer')
#         trainer.restore(config['checkpoint'])
#         print('done')
#
#     # trainer = rllib.agents.ppo.PPOTrainer(config=config, env=ThisRoadEnv)`
#     # trainer = rllib.agents.es.ESTrainer(config=config, env=ThisRoadEnv)
#     # trainer = ESTweakTrainer(config=config, env=ThisRoadEnv)
#     # trainer = ESActualTrainer(config=rllib_config, env=this_env, logger_creator=logger_creator)
#     # trainer = ESCOTrainer(config=rllib_config, env=this_env, logger_creator=logger_creator)
#     # trainer = ESCOTrainer(config=rllib_config, env=this_env)
#     # trainer = CoordinatedDPSTrainer(config=rllib_config, env=this_env, logger_creator=logger_creator)
#     return trainer

def make_trainer(rllib_config) -> Trainer:
    config = rllib_config['env_config']['config']  # unwrap config from rllib config
    config, this_env = setup_environment(config)
    
    results_path = DEFAULT_RESULTS_DIR
    checkpoint_path = results_path
    # checkpoint_path = os.path.join(results_path, config['group'])
    # checkpoint_path = os.path.join(checkpoint_path, config['name'])
    
    logdir_prefix = config['group'] + '_' + config['name'] + '_' + datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    #
    print('checkpoint_path:', checkpoint_path, 'results_path', results_path, 'logdir_prefix', logdir_prefix)
    
    def logger_creator(config):
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        logdir = tempfile.mkdtemp(
            prefix=logdir_prefix, dir=checkpoint_path)
        print('logger creator: ', logdir)
        return UnifiedLogger(config, logdir, None)
    
    trainer_setting = config['trainer']
    if trainer_setting == 'ESActualTrainer':
        trainer = ESActualTrainer(config=rllib_config, env=this_env)
    elif trainer_setting == 'ESCOTrainer':
        trainer = ESCOTrainer(config=rllib_config, env=this_env, logger_creator=logger_creator)
    elif trainer_setting == 'CoordinatedDPSTrainer':
        trainer = CoordinatedDPSTrainer(config=rllib_config, env=this_env, logger_creator=logger_creator)
    else:
        raise Exception('Unimplemented trainer "' + trainer_setting + '"')
    
    if 'checkpoint' in config and config['checkpoint'] is not None:
        print('restore trainer')
        trainer.restore(config['checkpoint'])
        print('done')
    
    # trainer = rllib.agents.ppo.PPOTrainer(config=config, env=ThisRoadEnv)`
    # trainer = rllib.agents.es.ESTrainer(config=config, env=ThisRoadEnv)
    # trainer = ESTweakTrainer(config=config, env=ThisRoadEnv)
    # trainer = ESActualTrainer(config=rllib_config, env=this_env, logger_creator=logger_creator)
    # trainer = ESCOTrainer(config=rmake_trainerllib_config, env=this_env, logger_creator=logger_creator)
    # trainer = ESCOTrainer(config=rllib_config, env=this_env)
    # trainer = CoordinatedDPSTrainer(config=rllib_config, env=this_env, logger_creator=logger_creator)
    return trainer


def train(rllib_config, reporter):
    ego_starting_distance = 600.0
    environment, trainer = make_environment_and_controller(None, rllib_config)
    # trainer = make_trainer(config)
    checkpoint_frequency = 1
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


print('begin trainer')

default_config = common_default_config

ray_num_cpus = None
if len(sys.argv) >= 4 and sys.argv[-3] == 'ray':
    redis_password = sys.argv[-2]
    ray_num_cpus = int(sys.argv[-1])
    ray.init(address=os.environ["ip_head"], _redis_password=redis_password)
    sys.argv = sys.argv[0:-3]
    # del sys.argv[-1:-4]
    print('ray configuration: ', redis_password, ray_num_cpus, 'argv: ', sys.argv)
else:
    if not ray.is_initialized():
        ray.init()

print('setup config')

config, run_prefix = setup_run(default_config)
# config, this_env = setup_environment_config(config)

print("Nodes in the Ray cluster:")
pprint(ray.nodes())
pprint(ray.cluster_resources())

if ray_num_cpus is not None:
    config['rllib']['num_workers'] = ray_num_cpus - 1

rllib_config = make_rllib_config(config)

print('running tune')

tune.run(
    train,
    name=config['name'],
    trial_name_creator=lambda trial: config['name'],
    config=rllib_config,
    # local_dir='~/ray_results'
    # resources_per_trial={'gpu':1},
    )

print('shutting down')
ray.shutdown()
print('done')