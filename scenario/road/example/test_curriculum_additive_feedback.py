import os

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
            road.RoadTerminator(time_limit=3 * 60),
            road.RoadGoalRewarder(),
            []
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
    # max_reward = 85
    # min_reward = 60
    target_reward = 90
    gain = 1.0
    decay = 0.0

    upper_limit = 600
    lower_limit = 10.0

    ego_starting_distance = 20.0

    checkpoint_frequency = 50
    max_timesteps = int(400e3)

    trainer = rllib.agents.ppo.PPOTrainer(config=config, env=ThisRoadEnv)

    # trainer = rllib.agents.es.ESTrainer(config=config, env=ThisRoadEnv)

    accumulator = target_reward

    def set_starting_distance(ego_starting_distance):
        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.process.set_starting_distance(ego_starting_distance)))

    # def set_starting_distance(ego_starting_distance):
    #     for worker in trainer._workers:
    #         print(worker)
    #         worker.env.process.set_starting_distance(ego_starting_distance)

    set_starting_distance(ego_starting_distance)
    i = 0
    go = True
    while go:
        result = trainer.train()
        reporter(**result)

        mean_reward = result['episode_reward_mean']
        signal = decay * accumulator + (1 - decay) * mean_reward
        difficulty_delta = gain * (signal - target_reward)
        # accumulator = accumulator - difficulty_delta
        ego_starting_distance = ego_starting_distance + difficulty_delta
        # ego_starting_distance = ego_starting_distance * (1 + difficulty_delta)
        # ego_starting_distance = ego_starting_distance * math.pow(gain, (signal - target_reward))

        ego_starting_distance = max(lower_limit, min(upper_limit, ego_starting_distance))
        print('mean reward ', mean_reward, ' using ego_starting_distance of ', ego_starting_distance)
        set_starting_distance(ego_starting_distance)

        go = result['timesteps_total'] < max_timesteps

        if not go or i % checkpoint_frequency == 0:
            checkpoint = trainer.save()
            print('saved to checkpoint ', checkpoint)
        i = i + 1


def on_episode_end(info):
    # print(info)
    episode = info['episode']
    # print(info)
    # trainer = info['trainer']
    base_env = info['env']
    episode.custom_metrics['ego_starting_distance'] = base_env.get_unwrapped()[0].process.ego_starting_distance


num_worker_cpus = 11
tune.run(
    train,
    name='curriculum_test_1',
    trial_name_creator=tune.function(lambda trial: 'additive_feedback_50target_4gain_0decay'),
    config={

        'num_gpus': 1,
        'num_workers': num_worker_cpus,
        'num_cpus_per_worker': 1,
        'num_gpus_per_worker': 1.0 / num_worker_cpus,
        'sample_batch_size': 200,
        'train_batch_size': int(2 * 60.0 / .05),
        'batch_mode': 'truncate_episodes',  # 'complete_episodes',  # 'truncate_episodes',
        'timesteps_per_iteration': int(2 * 60 / .05),
        'sgd_minibatch_size': 128,
        # 'shuffle_sequences':       True,
        'num_sgd_iter': 30,
        'gamma': 0.99999,
        'lr': 0.0001,
        'callbacks': {
            'on_episode_end': tune.function(on_episode_end),
        },
        'model': {
            'conv_filters': None,
            'fcnet_activation': 'relu',
            'fcnet_hiddens': [256, 256, 16, 16],
            # 'vf_share_layers': False,
        }
    },
)
