import time

import ray
import ray.rllib as rllib
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

import factored_gym as framework
import scenario.road as road
# from  cavs_environments.vehicle.scenario.road.road_occupancy_grid_observer_flattened import RoadOccupancyGridObserver
from scenario import PathPlanningModel
# from  cavs_environments.vehicle.scenario.road.road_process import RoadProcessOnCarlaMap
from scenario import PathPlanningProcess
from scenario import PathPlanningRewarder
from scenario import PathPlanningTerminator
from scenario.road.factored_road_observer import FactoredRoadObserver

### To tunnel to eagle for use of tensforboard: e.g.,
##  ssh -L 6006:127.0.0.1:6006 pgraf@el3.hpc.nrel.gov
###

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# ray.init(num_gpus=2)
if not ray.is_initialized():
    ray.init()


def env_creator(env_config):
    env = PathPlanningEnv(env_config)
    return env


register_env("PathPlanningEnv-v0", env_creator)
ModelCatalog.register_custom_model("path_planning_model", PathPlanningModel)

USE_FACTORED_OBS = True  ## triggers changes in env_config that are then parsed out


class PathPlanningEnv(framework.FactoredGym):

    def __init__(self, env_config=None):
        ## moving to env_config having sub parts for process, obs, reward,... but process a little behind and has
        ## still some direct entries in env_config, as of 2/21
        observation_scaling = 1.0  # 10.0
        ego_starting_distance = 2000.0

        print("env_config", env_config)
        pp_process_config = env_config['process_config'] if 'process_config' in env_config else {}

        if "use_factored_obs" in env_config and env_config['use_factored_obs']:
            obs_config = env_config['obs_config']
            observer = FactoredRoadObserver(obs_config)
        else:
            observer = road.RoadObserver(observation_scaling)
        #            RoadOccupancyGridObserver(observation_scaling, grid_length = 9, grid_width = 5),
        #            RoadOccupancyGridObserver(observation_scaling, grid_length = 18, grid_width = 10),  #, frames_to_observe=1),

        # IFEAGLE
        ### Mac vs Linux vs eagle!!!
        # linux
        scenario_file = "/home/pgraf/work/cavs/CarlaProjects/cavs-environments_roadmaker-clean/cavs_environments/vehicle/k_road/scenario/road/example/multi_route_scenario_town04_jct.json"
        # eagle
        #        scenario_file = "/home/pgraf/projects/cavs/cavs-environments_2_23_21/cavs_environments/vehicle/k_road/scenario/road/example/multi_route\
        # _scenario_town04_jct.json"
        # Mac
        #        scenario_file = "/Users/pgraf/work/CAVs/CarlaProjects/cavs_env21/cavs_environments/vehicle/k_road/scenario/road/example/multi_route_scenario_town04_jct.json"
        scenario = "cross_intersection"

        with_carla = False
        if "scenario" in env_config:
            scenario = env_config['scenario']
        if "with_carla" in env_config:
            with_carla = env_config['with_carla']

        if scenario == "curvy_road":
            bot_routes = None
            ego_routes = [2]
            traffic_density = 0
            target_inset = 20
            ego_starting_distance = 65.0

        elif scenario == "avoid_bots":
            bot_routes = [1]
            ego_routes = [1]
            traffic_density = 30
            target_inset = 5
            ego_starting_distance = 2000.0
        elif scenario == "cross_intersection":
            bot_routes = [1]
            ego_routes = [0]
            traffic_density = 12
            target_inset = 5
            ego_starting_distance = 2000.0

        # scenario_file = "/home/pgraf/work/cavs/CarlaProjects/carla-collect-2_28_20/./docs/scenarios/town01/scenario_009.json"   #13=has intersection, 9=turn, no intersection
        # process = RoadProcessOnCarlaMap(scenario_file, ego_starting_distance=ego_starting_distance, traffic_density=0.00, target_inset=15)  #0.02
        # process =road.RoadProcess(ego_starting_distance=ego_starting_distance),

        #        scenario_file = "/home/pgraf/work/cavs/CarlaProjects/cavs-environments_k_road_roadmaker/cavs_environments/vehicle/k_road/scenario/road/example/multi_route_scenario_000b.json"

        # Learned to drive this with lidar (even w/out vehicle state in obs)
        #        scenario_file = "/home/pgraf/work/cavs/CarlaProjects/cavs-environments_roadmaker-clean/cavs_environments/vehicle/k_road/scenario/road/example/multi_route_scenario_town01_all-nobndry.json"
        #        route_num = 7

        #        scenario_file = "/home/pgraf/work/cavs/CarlaProjects/cavs-environments_roadmaker-clean/cavs_environments/vehicle/k_road/scenario/road/example/multi_route_scenario_town04_merge.json"
        #        scenario_file = "/home/pgraf/work/cavs/CarlaProjects/cavs-environments_roadmaker-clean/cavs_environments/vehicle/k_road/scenario/road/example/multi_route_scenario_town04_local.json"
        route_num = 1  ## only draw this route even though there are more "paths"
        #        route_num = None   ## draws all routes, ego goes on 0, no control over which that is
        if not with_carla:
            process = PathPlanningProcess(scenario_file, traffic_density=traffic_density, speed_mean=0,
                                          target_inset=target_inset,
                                          ego_starting_distance=ego_starting_distance, ego_routes=ego_routes,
                                          bot_routes=bot_routes,
                                          pp_process_config=pp_process_config)
            ##speed_mean=0 makes bot vehicles stationary

        else:
            process = KRoadCarlaTwinRoadProcess(scenario_file, mode=KCModes.KC, traffic_density=traffic_density,
                                                speed_mean=8, target_inset=target_inset,
                                                ego_starting_distance=ego_starting_distance, ego_routes=ego_routes,
                                                bot_routes=bot_routes)
        #            process = KRoadCarlaTwinRoadProcess(scenario_file, mode=KCModes.K0, traffic_density=0.00, target_inset=0,
        #                    ego_starting_distance=ego_starting_distance, fixed_route_num = route_num)

        # speed_mean=12,
        # target_inset = 70

        terminator = PathPlanningTerminator()
        rewarder_config = {}
        if "rewarder_config" in env_config:
            rewarder_config = env_config['rewarder_config']
        rewarder = PathPlanningRewarder(rewarder_config)
        process.observer = observer

        super().__init__(
            process,
            observer,
            terminator,
            rewarder,
        )


def train(config, reporter):
    ego_starting_distance = 90.0
    print("CONFIG")
    print(config)

    checkpoint_frequency = 1
    max_iters = int(400e3)

    trainer = rllib.agents.ppo.PPOTrainer(config=config, env=PathPlanningEnv)
    if "env_config" in config:
        if 'checkpoint' in config['env_config'] and config['env_config']['checkpoint'] != None:
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

    def set_disable_steering(disable):
        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.process.set_disable_steering(disable)))

    # def set_starting_distance(ego_starting_distance):
    #     for worker in trainer._workers:
    #         print(worker)
    #         worker.env.process.set_starting_distance(ego_starting_distance)

    ## This has to match testing, which unless we set up something special has to just use default in env constructor, so
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


# PPO_CONFIG = {
#     "env": 'CarlaRoadEnv-v0',
#     "model": {
#         "custom_model": "carla_road_model",
#     },
#     "sample_batch_size": 400,
#     "num_workers": 15,
#     "num_gpus": 2,
#     "env_config": {"place_holder": 5, },
# }


def do_train(inp={}):
    obs_config = {'odometry': {},
                  #                    'lidar' : {'forward_scan_resolution':20, 'rear_scan_resolution':20, 'lidar_channels':2}  #,
                  'occupancy_grid': {'grid_dims': [36, 20, 2],
                                     'cnn_structure': [[16, [5, 5], (2, 2)],
                                                       [32, [3, 3], (2, 2)],
                                                       [64, [5, 3], (1, 1)]]}
                  }
    #  'occupancy_grid':{  'grid_dims' : [18,10,2],
    #                      'cnn_structure':    [[16, [5, 5], (2,2)],
    #                                          [32, [3,3], (2,2)],
    #                                          [64, [5,3], (1,1)]]}
    #                                         }

    rewarder_config = {'collision_bonus': -3, 'distance_bonus': 2, 'completion_bonus': 20}
    process_config = {"min_path_length": 30}

    if 'rewarder_config' in inp:
        rewarder_config = inp['rewarder_config']
    if 'process_config' in inp:
        process_config = inp['process_config']

    checkpoint = None
    #    checkpoint = "/home/pgraf/ray_results/PPO_CarlaRoadEnv_2020-06-21_06-34-07jli_z1a8/checkpoint_1841/checkpoint-1841"
    #    checkpoint = "/home/pgraf/ray_results/PPO_CarlaRoadEnv_2020-06-23_15-55-324fhtylgu/checkpoint_1556/checkpoint-1556"
    #    checkpoint = "/home/pgraf/ray_results/PPO_CarlaRoadEnv_2020-06-26_11-32-37v105jwps/checkpoint_51/checkpoint-51"

    ##   checkpoint = "/home/pgraf/ray_results/cross_intersection-no_stearing_test1/checkpoint_611/checkpoint-611"

    if "grid_dims" in inp:
        obs_config['occupancy_grid']['grid_dims'] = inp['grid_dims']
    scenario = "avoid_bots"  ## one of: cross_intersection, curvy_road, avoid_bots
    if "scenario" in inp:
        scenario = inp['scenario']
    name = 'path_planning'
    if "name" in inp:
        name = inp['name']

    with_carla = False
    if USE_FACTORED_OBS:
        model_config = {
            "custom_model": "path_planning_model",
            "custom_model_config": {'obs_config': obs_config}
        }
        env_config = {'obs_config': obs_config, "checkpoint": checkpoint, "use_factored_obs": True,
                      "scenario": scenario, "with_carla": with_carla,
                      'rewarder_config': rewarder_config, 'process_config': process_config}
    else:
        model_config = {
            #                'conv_filters':      [[16, [3, 3], 2], [32, [3,2],2], [64, [3,2], 1]],  ## valid for grid=9x5
            #                'conv_filters':      [[16, [5, 5], 2], [32, [3,3],2], [64, [5,3], 1]],  ## valid for grid=18x10
            #                "use_lstm": True,     ## PG ??
            # This gets set to False somewhere anyway               'vf_share_layers': True,
            #                'conv_filters':      [[16, [3, 3], 1], [32, [3, 3], 1], [128, [3, 3], 1]],
            'fcnet_activation': 'relu',
            'fcnet_hiddens': [256, 256, 64, 64],
        }
        env_config = {"use_factored_obs": False, "scenario": scenario, "with_carla": with_carla}

    num_worker_cpus = 11  # IFEAGLE

    tune.run(
        train,
        name=name,
        # local_dir="/scratch/pgraf/ray_results",   #IFEAGLE
        trial_name_creator=tune.function(lambda trial: 'standard_' + str(time.time())),
        config={

            # xz            'num_gpus':                2,
            'num_workers': num_worker_cpus,
            #        'num_cpus_per_worker':     1,
            #        'num_gpus_per_worker':     0,  #1.0 / num_worker_cpus,
            ##            'sample_batch_size':       200,
            ##            'train_batch_size':        int(2 * 60.0 / .05),
            ##            'batch_mode':              'truncate_episodes',  # 'complete_episodes',  # 'truncate_episodes',
            ##            'timesteps_per_iteration': int(2 * 60 / .05),
            ##            'sgd_minibatch_size':      128,
            # 'shuffle_sequences':       True,
            ##            'num_sgd_iter':            30,
            ##            'gamma':                   0.9999,
            ##            'lr':                      0.0001,
            ##   'entropy_coeff':            0.0,   ### PG ??
            'callbacks': {
                'on_episode_end': tune.function(on_episode_end),
            },
            "env": 'PathPlanningEnv-v0',
            'model': model_config,
            'env_config': env_config
        }
    )


# (pid=2875) LAST LAYER  Tensor("default_policy/conv_value_1/Relu:0", shape=(?, 21, 21, 16), dtype=float32) 16 [4, 4] 2
# (pid=2875) LAST LAYER  Tensor("default_policy/conv_value_2/Relu:0", shape=(?, 11, 11, 32), dtype=float32) 32 [4, 4] 2
# (pid=2875) LAST LAYER 2 Tensor("default_policy/conv_value_3/Relu:0", shape=(?, 1, 1, 256), dtype=float32) 256 [11, 11] 1
# (pid=2875) LAST LAYER 3  Tensor("default_policy/conv_value_out/BiasAdd:0", shape=(?, 1, 1, 1), dtype=float32)

# (pid=3149) LAST LAYER  Tensor("default_policy/conv_value_1/Relu:0", shape=(?, 9, 5, 16), dtype=float32) 16 [3, 3] 1
# (pid=3149) LAST LAYER  Tensor("default_policy/conv_value_2/Relu:0", shape=(?, 9, 5, 32), dtype=float32) 32 [3, 2] 1
# (pid=3149) LAST LAYER 2 Tensor("default_policy/conv_value_3/Relu:0", shape=(?, 5, 4, 64), dtype=float32) 64 [5, 2] 1
# (pid=3149) LAST LAYER 3  Tensor("default_policy/conv_value_out/BiasAdd:0", shape=(?, 5, 4, 1), dtype=float32)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inpfile",
        type=str,
        default=None)
    args = parser.parse_args()

    if args.inpfile != None:
        inp = json.load(args.inpfile)
    else:
        inp = {}

    do_train(inp)
