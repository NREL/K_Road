import math
import os
import time

import ray
import ray.rllib as rllib
from ray import tune

import scenario.road as road

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

ray.init(num_gpus=1)


class ThisRoadEnv(factored_gym.FactoredGym):

    def __init__(self, env_config):
        observation_scaling = 1.0  # 10.0
        ego_starting_distance = 5.0
        super().__init__(
            road.RoadProcess(ego_starting_distance=ego_starting_distance),
            road.RoadObserver(observation_scaling),
            road.RoadTerminator(time_limit=5 * 60),
            road.RoadGoalRewarder(),
            [factored_gym.ActionCenterer([.001, 5], [0, 0])]
        )


# class ThisRoadEnv(framework.FactoredGym):
#
#     def __init__(self, env_config):
#         observation_scaling = 1.0  # 10.0
#         action_scaling = 1.0  # 1.0 / 10.0
#         ego_starting_distance = 5.0
#         super().__init__(
#             road.RoadProcess(ego_starting_distance=ego_starting_distance),
#             road.RoadObserver(observation_scaling),
#             road.RoadTerminator(),
#             road.RoadGoalRewarder(),
#             [framework.ActionScaler(action_scaling), framework.ActionCenterer([.1, 1], [-.5, 0])]
#         )


def train(config, reporter):
    max_reward = 98
    min_reward = 50
    delta = 5.0
    decay = 0.0

    upper_limit = 600
    lower_limit = 10.0

    ego_starting_distance = 10.0

    checkpoint_frequency = 50
    max_iters = int(400e3)

    trainer = rllib.agents.ppo.PPOTrainer(config=config, env=ThisRoadEnv)

    # trainer = rllib.agents.es.ESTrainer(config=config, env=ThisRoadEnv)

    accumulator = (min_reward + max_reward) / 2.0

    def set_starting_distance(ego_starting_distance):
        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.process.set_starting_distance(ego_starting_distance)))

    # def set_starting_distance(ego_starting_distance):
    #     for worker in trainer._workers:
    #         print(worker)
    #         worker.env.process.set_starting_distance(ego_starting_distance)

    set_starting_distance(ego_starting_distance)
    for i in range(max_iters):
        result = trainer.train()
        reporter(**result)

        if i % checkpoint_frequency == 0:
            checkpoint = trainer.save()
            print('saved to checkpoint ', checkpoint)

        mean_reward = result['episode_reward_mean']
        accumulator = decay * accumulator + (1 - decay) * mean_reward
        if accumulator >= max_reward:
            difficulty_delta = delta
        elif accumulator <= min_reward:
            difficulty_delta = -delta
        else:
            difficulty_delta = 0.0

        ego_starting_distance = ego_starting_distance + difficulty_delta

        ego_starting_distance = max(lower_limit, min(upper_limit, ego_starting_distance))
        print('mean reward ', mean_reward, ' using ego_starting_distance of ', ego_starting_distance)
        set_starting_distance(ego_starting_distance)


def on_episode_end(info):
    # print(info)
    episode = info['episode']
    # print(info)
    # trainer = info['trainer']
    base_env = info['env']
    episode.custom_metrics['ego_starting_distance'] = base_env.get_unwrapped()[0].process.ego_starting_distance


# def test(config, path):
#     trainer = rllib.agents.ppo.PPOTrainer(config=config, env=ThisRoadEnv)
#     trainer.restore()

num_gpus = 1
num_worker_cpus = 10
cpus_per_worker = 1
num_workers = math.floor(num_worker_cpus / cpus_per_worker)
gpu_per_worker = 1.0 / num_workers

config = {
    'num_gpus': num_gpus,
    'num_workers': num_workers,
    'num_cpus_per_worker': cpus_per_worker,
    'num_gpus_per_worker': gpu_per_worker,
    'sample_batch_size': 200,
    'train_batch_size': int(2 * 60.0 / .05),
    'batch_mode': 'truncate_episodes',  # 'complete_episodes',  # 'truncate_episodes',
    'timesteps_per_iteration': int(2 * 60 / .05),
    'sgd_minibatch_size': 128,
    # 'shuffle_sequences':       True,
    'num_sgd_iter': 30,
    'gamma': 0.9999,
    'lr': 0.0001,
    'callbacks': {
        'on_episode_end': tune.function(on_episode_end),
    },
    'model': {
        'conv_filters': None,
        'fcnet_activation': 'relu',
        'fcnet_hiddens': [256, 128, 64, 32, 16, 8],
        # 'vf_share_layers': False,
    },
    # 'monitor':                 True
}

# test(config, '/home/ctripp/ray_results/PPO_ThisRoadEnv_2019-08-22_21-10-247wdvjdm_/checkpoint_251/checkpoint-251')


tune.run(
    train,
    name='curriculum_test_1',
    trial_name_creator=tune.function(lambda trial: 'threshold_10m_80max_50min_' + str(time.time())),
    config=config,
)
