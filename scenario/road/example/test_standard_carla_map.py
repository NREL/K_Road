import time

import ray
import ray.rllib as rllib
from ray import tune

import scenario.road as road
# from  cavs_environments.vehicle.scenario.road.road_process import RoadProcessOnCarlaMap
from scenario import KRoadProcessOnCarlaMap
# from  cavs_environments.vehicle.scenario.road.road_occupancy_grid_observer_flattened import RoadOccupancyGridObserver
from scenario.road import RoadOccupancyGridObserver

# from  cavs_environments.vehicle.scenario.road.road_process import RoadProcessOnCarlaMap

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ray.init(num_gpus=0)
if not ray.is_initialized():
    ray.init()


class CarlaRoadEnv(factored_gym.FactoredGym):

    def __init__(self, env_config=None):
        observation_scaling = 1.0  # 10.0
        ego_starting_distance = 30.0

        # scenario_file = "/home/pgraf/work/cavs/CarlaProjects/carla-collect-2_28_20/./docs/scenarios/town01/scenario_009.json"   #13=has intersection, 9=turn, no intersection
        # process = RoadProcessOnCarlaMap(scenario_file, ego_starting_distance=ego_starting_distance, traffic_density=0.00, target_inset=15)  #0.02
        # process =road.RoadProcess(ego_starting_distance=ego_starting_distance),

        #        scenario_file = "/home/pgraf/work/cavs/CarlaProjects/cavs-environments_k_road_roadmaker/cavs_environments/vehicle/k_road/scenario/road/example/multi_route_scenario_000b.json"

        # Learned to drive this with lidar (even w/out vehicle state in obs)
        scenario_file = "/home/pgraf/work/cavs/CarlaProjects/cavs-environments_roadmaker-clean/cavs_environments/vehicle/k_road/scenario/road/example/multi_route_scenario_town01_all.json"
        route_num = 7

        #        scenario_file = "/home/pgraf/work/cavs/CarlaProjects/cavs-environments_roadmaker-clean/cavs_environments/vehicle/k_road/scenario/road/example/multi_route_scenario_town04_merge.json"
        #        route_num = 0
        process = KRoadProcessOnCarlaMap(scenario_file, traffic_density=0.00, target_inset=70,
                                         ego_starting_distance=ego_starting_distance, fixed_route_num=route_num)
        # process = KRoadCarlaTwinRoadProcess(scenario_file, mode=KCModes.K0, traffic_density=0.00, target_inset=0, 
        #     ego_starting_distance=ego_starting_distance, fixed_route_num = route_num)

        # speed_mean=12,

        super().__init__(
            process,
            #            road.RoadObserver(observation_scaling),
            RoadOccupancyGridObserver(observation_scaling, grid_length=9, grid_width=5),
            #            RoadOccupancyGridObserver(observation_scaling, grid_length = 18, grid_width = 10),  #, frames_to_observe=1),
            road.RoadTerminator(time_limit=60),  # (time_limit=3 * 60),
            road.RoadRewarder(),
            # road.RoadOccupancyGridRewarder()
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
    ego_starting_distance = 90.0
    print("CONFIG")
    print(config)

    checkpoint_frequency = 5
    max_iters = int(400e3)

    trainer = rllib.agents.ppo.PPOTrainer(config=config, env=CarlaRoadEnv)
    if "env_config" in config:
        if 'checkpoint' in config['env_config']:
            trainer.restore(config['env_config']['checkpoint'])
            print("restored from", config['env_config']['checkpoint'])

    if "env_config" in config:
        if 'render' in config['env_config']:
            trainer.render = config['env_config']['render']

    # trainer = rllib.agents.es.ESTrainer(config=config, env=ThisRoadEnv)

    def set_starting_distance(ego_starting_distance):
        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.process.set_starting_distance(ego_starting_distance)))

    def set_route(route_num):
        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.process.set_route(route_num)))

    # def set_starting_distance(ego_starting_distance):
    #     for worker in trainer._workers:
    #         print(worker)
    #         worker.env.process.set_starting_distance(ego_starting_distance)

    ## This has to match testing, which unless we set up something special has to just use default in env
    # constructor, so
    # ignore this for now.
    # set_starting_distance(ego_starting_distance)

    for i in range(max_iters):
        #  set_route(None)  ## None lets road pick a route randomly
        result = trainer.train()
        reporter(**result)

        if i % checkpoint_frequency == 0:
            checkpoint = trainer.save()
            print('saved to checkpoint ', checkpoint)


def on_episode_end(info):
    # print(info)
    episode = info['episode']
    # print(info)
    # trainer = info['trainer']
    base_env = info['env']
    episode.custom_metrics['ego_starting_distance'] = base_env.get_unwrapped()[0].process.ego_starting_distance


def do_train():
    num_worker_cpus = 11
    tune.run(
        train,
        name='curriculum_test_1',
        trial_name_creator=tune.function(lambda trial: 'standard_' + str(time.time())),
        config={

            'num_gpus': 1,
            'num_workers': num_worker_cpus,
            #        'num_cpus_per_worker':     1,
            #        'num_gpus_per_worker':     0,  #1.0 / num_worker_cpus,
            'sample_batch_size': 200,
            'train_batch_size': int(2 * 60.0 / .05),
            'batch_mode': 'truncate_episodes',  # 'complete_episodes',  # 'truncate_episodes',
            'timesteps_per_iteration': int(2 * 60 / .05),
            'sgd_minibatch_size': 128,
            # 'shuffle_sequences':       True,
            'num_sgd_iter': 30,
            'gamma': 0.9999,
            'lr': 0.0001,
            ##   'entropy_coeff':            0.0,   ### PG ??
            'callbacks': {
                'on_episode_end': tune.function(on_episode_end),
            },
            'model': {
                'conv_filters': [[16, [3, 3], 2], [32, [3, 2], 2], [64, [3, 2], 1]],  ## valid for grid=9x5
                #                'conv_filters':      [[16, [5, 5], 2], [32, [3,3],2], [64, [5,3], 1]],  ## valid for grid=18x10
                "use_lstm": True,  ## PG ??
                # This gets set to False somewhere anyway               'vf_share_layers': True,
                #                'conv_filters':      [[16, [3, 3], 1], [32, [3, 3], 1], [128, [3, 3], 1]],
                'fcnet_activation': 'relu',
                'fcnet_hiddens': [256, 256, 64, 64],
            }

        },
    )


# (pid=2875) LAST LAYER  Tensor("default_policy/conv_value_1/Relu:0", shape=(?, 21, 21, 16), dtype=float32) 16 [4, 4] 2
# (pid=2875) LAST LAYER  Tensor("default_policy/conv_value_2/Relu:0", shape=(?, 11, 11, 32), dtype=float32) 32 [4, 4] 2
# (pid=2875) LAST LAYER 2 Tensor("default_policy/conv_value_3/Relu:0", shape=(?, 1, 1, 256), dtype=float32) 256 [11,
# 11] 1
# (pid=2875) LAST LAYER 3  Tensor("default_policy/conv_value_out/BiasAdd:0", shape=(?, 1, 1, 1), dtype=float32)

# (pid=3149) LAST LAYER  Tensor("default_policy/conv_value_1/Relu:0", shape=(?, 9, 5, 16), dtype=float32) 16 [3, 3] 1
# (pid=3149) LAST LAYER  Tensor("default_policy/conv_value_2/Relu:0", shape=(?, 9, 5, 32), dtype=float32) 32 [3, 2] 1
# (pid=3149) LAST LAYER 2 Tensor("default_policy/conv_value_3/Relu:0", shape=(?, 5, 4, 64), dtype=float32) 64 [5, 2] 1
# (pid=3149) LAST LAYER 3  Tensor("default_policy/conv_value_out/BiasAdd:0", shape=(?, 5, 4, 1), dtype=float32)


### This works, but I can't for the life of me figure out how to get it to rendr !!! ###
def do_test():
    checkpoint = "/Users/pgraf/ray_results/PPO_ThisRoadEnv_2020-05-13_12-10-58f2svb2cu/checkpoint_101/checkpoint-101"
    num_worker_cpus = 0
    tune.run(
        train,
        name='curriculum_test_1',
        trial_name_creator=tune.function(lambda trial: 'standard_' + str(time.time())),
        config={
            "explore": False,
            #       'num_gpus':                0,
            'num_workers': num_worker_cpus,
            #        'num_cpus_per_worker':     1,
            #        'num_gpus_per_worker':     0,  #1.0 / num_worker_cpus,
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
                # 'conv_filters':      [[16, [3, 3], 1], [32, [3, 3], 1], [128, [3, 3], 1]],
                'fcnet_activation': 'relu',
                'fcnet_hiddens': [256, 256, 64, 64],
                # 'vf_share_layers': False,
            },
            "env_config": {'render': True, 'checkpoint': checkpoint}
        },
    )


if __name__ == "__main__":
    do_train()

# "model": {"dim": 42, "conv_filters": [[16, [4, 4], 2], [32, [4, 4], 2], [512, [11, 11], 1]]}
