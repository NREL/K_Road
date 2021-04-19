# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.

from __future__ import (
    absolute_import,
    division,
    print_function,
)

from typing import Optional

from ray.rllib.utils import try_import_tf

from coodinated_optimizer.NoiseTable import NoiseTable
from coodinated_optimizer.coordinated_optimizer import CoordinatedOptimizer
from coodinated_optimizer.optimizer_factory import get_optimizer_factory

tf = try_import_tf()

from ray.rllib.agents.es import ESTFPolicy
from ray.rllib.agents.es.es_tf_policy import rollout
from ray.rllib.utils import try_import_tf

tf = try_import_tf()

# from ray.rllib.agents.es import policies

# from ray.rllib.agents.es import policies

import asyncio
import math

from ray.rllib.utils import try_import_tf

from . import utils

tf = try_import_tf()

import logging
import numpy as np
import time

import ray
from ray.rllib.agents import (
    Trainer,
    with_common_config,
)

# from ray.rllib.agents.es import policies
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.annotations import override
from ray.rllib.utils import FilterManager

logger = logging.getLogger(__name__)

# yapf: disable
# __sphinx_doc_begin__
DEFAULT_CONFIG = with_common_config({
    # 'theta_decay':              0.001,
    # 'alpha':                    .2,
    # 'noise_stdev':              0.02,
    # 'candidates_per_iteration': 144,
    'timestep_limit': None,
    'num_evals_per_iteration': 1,
    # 'return_proc_mode':   'centered_rank',
    'num_workers': 4,
    'request_interleaving': 2,
    # 'stepsize': 0.01,
    # 'observation_filter':      'MeanStdFilter',
    'noise_size': 33554432,
    'random_seed': None,
    # 'report_length':      10,
    "action_noise_std": 0.0,
})


# __sphinx_doc_end__
# yapf: enable


class Common:

    def __init__(
            self,
            config,
            environment_factory,
            optimizer_factory,
            noise_table: Optional[NoiseTable],
            theta=None,
    ) -> None:
        # policy_params = {"action_noise_std": 0.0}

        self.env = environment_factory(config["env_config"])
        from ray.rllib import models

        self.preprocessor = models.ModelCatalog.get_preprocessor(self.env)

        self.sess = utils.make_session(single_threaded=True)
        policy_cls = Common.get_policy_class(config)
        self.policy = policy_cls(
            self.env.observation_space,
            self.env.action_space,
            config)

        theta = self.policy.get_flat_weights() if theta is None else theta
        self.optimizer: CoordinatedOptimizer = optimizer_factory(
            theta,
            noise_table=noise_table,
            **config['env_config']['config']['optimizer'])

    @staticmethod
    def get_policy_class(config):
        if config["framework"] == "torch":
            from ray.rllib.agents.es.es_torch_policy import ESTorchPolicy

            policy_cls = ESTorchPolicy
        else:
            policy_cls = ESTFPolicy
        return policy_cls


@ray.remote
class Worker(object):

    def __init__(self,
                 config: dict,
                 environment_factory,
                 optimizer_factory,
                 theta,
                 noise_table: Optional[NoiseTable],
                 ) -> None:
        self.config = config
        self.common = Common(config, environment_factory, optimizer_factory, noise_table, theta)
        self.timestep_limit = config['timestep_limit']

    @property
    def filters(self):
        return {DEFAULT_POLICY_ID: self.common.policy.observation_filter}

    def sync_filters(self, new_filters):
        for k in self.filters:
            self.filters[k].sync(new_filters[k])

    def get_filters(self, flush_after=False):
        return_filters = {}
        for k, f in self.filters.items():
            return_filters[k] = f.as_serializable()
            if flush_after:
                f.clear_buffer()
        return return_filters

    def evaluate(self, candidate):
        # ******************************* how to evaluate a candidate message

        weights = self.common.optimizer.expand(candidate)
        self.common.policy.set_flat_weights(weights)

        rewards, length = \
            rollout(
                self.common.policy,
                self.common.env,
                timestep_limit=self.timestep_limit,
                add_noise=False)
        # if candidate == 0:
        logger.info('candidate {} {} {} {} {}'.format(
            candidate, weights[0], self.common.policy.get_flat_weights()[0], rewards.sum(), length))

        return rewards.sum(), length

    def tell(self, candidate_evals):
        self.common.optimizer.tell_messages(candidate_evals)


class CoordinatedDPSTrainer(Trainer):
    _name = 'CDPS'
    _default_config = DEFAULT_CONFIG

    @override(Trainer)
    def _init(
            self,
            config,
            environment_factory,
    ) -> None:
        if config['random_seed'] is None:
            config['random_seed'] = int(time.time())

        self.config = config

        optimizer_factory = get_optimizer_factory(config['env_config']['config']['optimizer']['name'])
        self.common = Common(config, environment_factory, optimizer_factory, None)

        self.num_evals_per_iteration: int = config['num_evals_per_iteration']
        self.request_interleaving: int = config['request_interleaving']

        # Create workers
        logger.info('Creating workers.')
        theta_id = ray.put(self.common.optimizer.best_candidate())
        noise_table_id = ray.put(self.common.optimizer.noise_table)
        self._workers = [
            Worker.remote(config, environment_factory, optimizer_factory, theta_id, noise_table_id)
            for _ in range(config['num_workers'])
        ]

        self.episodes_so_far = 0
        self.tstart = time.time()
        self.episode_reward_mean = None
        self.episode_len_mean = None

    @override(Trainer)
    def _train(self):
        optimizer: CoordinatedOptimizer = self.common.optimizer
        candidates = optimizer.ask()
        best_message = optimizer.best()
        for i in range(self.num_evals_per_iteration):
            candidates.append(best_message)

        num_candidates = len(candidates)
        dispatch_index: int = num_candidates - 1

        async def manage_worker(worker):
            nonlocal candidates, dispatch_index, result
            while dispatch_index >= 0:
                index = dispatch_index
                dispatch_index -= 1
                candidate = candidates[index]

                score, length = await worker.evaluate.remote(candidate)
                candidates[index] = (score, candidate, length)

        async def run_workers():
            workers = self._workers
            num_workers = len(workers)
            tasks = [asyncio.create_task(manage_worker(workers[i % num_workers]))
                     for i in range(self.request_interleaving * num_workers)]
            for task in tasks:  # synchronize on evaluations completed
                await task

        loop = asyncio.new_event_loop()
        loop.run_until_complete(run_workers())

        theta_evals = candidates[-self.num_evals_per_iteration:]
        candidate_evals = candidates[0:-self.num_evals_per_iteration]

        # update all workers
        update_completions = [worker.tell.remote(candidate_evals) for worker in self._workers]
        optimizer.tell_messages(candidate_evals)

        self.common.policy.set_flat_weights(optimizer.best_candidate())

        # synchronize on updates completed
        for completion in update_completions:
            ray.get(completion)

        episodes_this_iteration = len(candidate_evals)
        self.episodes_so_far += episodes_this_iteration

        # Now sync the filters
        FilterManager.synchronize({
            DEFAULT_POLICY_ID: self.common.policy.observation_filter
        }, self._workers)
        # FilterManager.synchronize({
        #     DEFAULT_POLICY_ID: self.common.policy.get_filter()
        #     }, self._workers)

        info = optimizer.status()

        def extract_columns(evaluations):
            rewards = np.fromiter((evaluation[0] for evaluation in evaluations), dtype=np.float32)
            lengths = np.fromiter((evaluation[2] for evaluation in evaluations), dtype=np.int)
            return rewards, lengths

        theta_rewards, theta_lengths = extract_columns(theta_evals)
        candidate_rewards, candidate_lengths = extract_columns(candidate_evals)

        def impute(func, a, min_size=1):
            return func(a) if a.size >= min_size else math.nan

        def accumulate_distribution_stats(name, a):
            nonlocal info
            info[name + 'mean'] = impute(np.mean, a)
            info[name + 'stdev'] = impute(np.std, a, 2)
            info[name + 'min'] = impute(np.min, a)
            info[name + 'max'] = impute(np.max, a)

        accumulate_distribution_stats('candidate_reward_', candidate_rewards)
        accumulate_distribution_stats('candidate_length_', candidate_lengths)

        accumulate_distribution_stats('best_reward_', theta_rewards)
        accumulate_distribution_stats('best_length_', theta_lengths)

        info['episodes_this_iter'] = episodes_this_iteration
        info['episodes_so_far'] = self.episodes_so_far

        result = {
            'episode_reward_mean': info['best_reward_mean'],
            'episode_len_mean': info['best_length_mean'],
            'timesteps_this_iter': np.sum(candidate_lengths),
            'episode_reward_max': info['best_reward_max'],
            'episode_reward_min': info['best_reward_min'],
            'episodes_this_iter': episodes_this_iteration,
            'episodes_total': self.episodes_so_far,
            'info': info,
        }

        return result

    @override(Trainer)
    def compute_action(self, observation, *args, **kwargs):
        action = self.common.policy.compute_actions(observation, update=False)[0]
        if kwargs.get("full_fetch"):
            return action, [], {}
        return action
        # return self.policy.compute(observation, update=False)[0]

    @override(Trainer)
    def _stop(self):
        # workaround for https://github.com/ray-project/ray/issues/1516
        for w in self._workers:
            w.__ray_terminate__.remote()

    def __getstate__(self):
        return {
            'weights': self.common.policy.get_flat_weights(),
            'filter': self.common.policy.observation_filter,
            'episodes_so_far': self.episodes_so_far,
        }

    def __setstate__(self, state):
        self.episodes_so_far = state['episodes_so_far']
        self.common.policy.set_flat_weights(state['weights'])
        self.common.policy.observation_filter = state["filter"]
        FilterManager.synchronize({
            DEFAULT_POLICY_ID: self.common.policy.observation_filter
        }, self._workers)
        # self.policy.set_weights(state['weights'])
        # self.policy.set_filter(state['filter'])
        # FilterManager.synchronize({
        #     DEFAULT_POLICY_ID: self.policy.get_filter()
        #     }, self._workers)
