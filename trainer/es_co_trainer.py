# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.

from __future__ import (
    absolute_import,
    division,
    print_function,
)

import asyncio
import math
import sys
from random import randint

from ray.rllib.agents.es import ESTFPolicy
from ray.rllib.agents.es.es_tf_policy import rollout
from ray.rllib.utils import try_import_tf

from . import utils
from .utils import (
    batched_weighted_sum,
    compute_wierstra_ranks,
)

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
    'theta_decay': 0.001,
    'alpha': .2,
    'noise_stdev': 0.02,
    'candidates_per_iteration': 144,
    'timestep_limit': None,
    'num_evals_per_iteration': 2,
    # 'return_proc_mode':   'centered_rank',
    'num_workers': 4,
    # 'stepsize': 0.01,
    'observation_filter': 'MeanStdFilter',
    'noise_size': 67108864,
    'random_seed': None,
    "action_noise_std": 0.0,
    # 'report_length':      10,
})


# __sphinx_doc_end__
# yapf: enable


class NoiseTable(object):

    def __init__(self, count, seed):
        self.noise = np.random.RandomState(seed).randn(count).astype(np.float32)

    def get(self, i, dim):
        assert dim <= len(self.noise)
        offset = i % (len(self.noise) - dim)
        return self.noise[offset:offset + dim]

    def sample_index(self, dim):
        return np.random.randint(0, len(self.noise) - dim + 1)


class ModelKeeper:

    def __init__(
            self,
            theta,
            alpha,
            theta_decay,
            noise_stdev,
            noise_table: NoiseTable
    ) -> None:
        self.theta = theta
        self.alpha = alpha
        self.theta_decay = theta_decay
        self.noise_stdev = noise_stdev
        self.noise_table: NoiseTable = noise_table

    @staticmethod
    def make_from_config(theta, config):
        return ModelKeeper(
            theta,
            config['alpha'],
            config['theta_decay'],
            config['noise_stdev'],
            NoiseTable(config['noise_size'], config['random_seed']))

    def update(self, evaluations):
        # evaluations.sort(key=lambda evaluation: evaluation[1]) # group same offsets together

        num_evaluations = len(evaluations)
        num_pairs = math.floor(num_evaluations / 2)
        shape = (num_pairs, 2)
        scores = np.fromiter((evaluation[1] for evaluation in evaluations), dtype=np.float32).reshape(shape)
        ranks = compute_wierstra_ranks(scores)

        noise_indices = (evaluations[2 * i][0][0] for i in range(num_pairs))
        g, count = batched_weighted_sum(
            ranks[:, 0] - ranks[:, 1],
            (self.noise_table.get(index, len(self.theta)) for index in noise_indices),
            batch_size=500)

        g *= self.alpha / (num_evaluations * self.noise_stdev)

        # Compute the new weights theta.
        self.theta = self.theta * (1.0 - self.theta_decay) + g
        # print('update g', np.square(g).sum(), np.square(self.theta).sum(), self.theta[0])
        return {
            'weights_norm': np.square(self.theta).sum(),
            'grad_norm': np.square(g).sum(),
        }

    def get_perturbed_weights(self, noise_index, multiplier):
        perturbation = multiplier * self.noise_stdev * self.noise_table.get(noise_index, len(self.theta))
        return self.theta + perturbation


class Common:

    def __init__(self, config, env_creator):

        self.env = env_creator(config["env_config"])
        from ray.rllib import models

        self.preprocessor = models.ModelCatalog.get_preprocessor(self.env)

        self.sess = utils.make_session(single_threaded=True)
        policy_cls = Common.get_policy_class(config)
        self.policy = policy_cls(self.env.observation_space,
                                 self.env.action_space, config)

        self.model_keeper = ModelKeeper(
            self.policy.get_flat_weights(),
            config['alpha'],
            config['theta_decay'],
            config['noise_stdev'],
            NoiseTable(config['noise_size'], config['random_seed']))

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
                 config,
                 env_creator,
                 theta,
                 ) -> None:
        self.config = config
        self.common = Common(config, env_creator)
        self.common.model_keeper.theta = theta
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
        noise_index, multiplier = candidate

        weights = self.common.model_keeper.get_perturbed_weights(noise_index, multiplier)
        self.common.policy.set_flat_weights(weights)

        rewards, length = \
            rollout(
                self.common.policy,
                self.common.env,
                timestep_limit=self.timestep_limit,
                add_noise=False)
        return rewards.sum(), length

    def update(self, candidate_evals):
        self.common.model_keeper.update(candidate_evals)


class ESCOTrainer(Trainer):
    _name = 'ESCO'
    _default_config = DEFAULT_CONFIG

    @override(Trainer)
    def _init(
            self,
            config,
            env_creator,
    ) -> None:
        if config['random_seed'] is None:
            config['random_seed'] = int(time.time())

        self.config = config
        self.candidates_per_iteration: int = config['candidates_per_iteration']
        self.common = Common(config, env_creator)
        self.num_evals_per_iteration = config['num_evals_per_iteration']

        # Create workers
        logger.info('Creating workers.')
        theta = self.common.policy.get_flat_weights()
        self._workers = [
            Worker.remote(config, env_creator, theta)
            for _ in range(config['num_workers'])
        ]

        self.episodes_so_far = 0
        self.tstart = time.time()
        self.episode_reward_mean = None
        self.episode_len_mean = None

    @override(Trainer)
    def _train(self):
        num_pairs = math.ceil(self.candidates_per_iteration / 2)
        candidates: [any] = [((0, 0), None, None)] * self.num_evals_per_iteration
        for _ in range(num_pairs):
            offset = randint(0, sys.maxsize)
            candidates.append(((offset, 1), None, None))
            candidates.append(((offset, -1), None, None))

        num_candidates = len(candidates)
        dispatch_index: int = 0

        async def manage_worker(worker):
            nonlocal candidates, dispatch_index, result
            while dispatch_index < num_candidates:
                index = dispatch_index
                dispatch_index += 1

                candidate = candidates[index][0]

                score, length = await worker.evaluate.remote(candidate)
                candidates[index] = (candidate, score, length)

        async def run_workers():
            tasks = [asyncio.create_task(manage_worker(worker)) for worker in self._workers]
            for task in tasks:
                await task

        print('run workers begin')
        loop = asyncio.new_event_loop()
        loop.run_until_complete(run_workers())
        print('run workers complete')
        # synchronize on evaluations completed

        theta_evals = candidates[0:self.num_evals_per_iteration]
        candidate_evals = candidates[self.num_evals_per_iteration:]

        # update all workers
        update_completions = [worker.update.remote(candidate_evals) for worker in self._workers]

        info = self.common.model_keeper.update(candidate_evals)
        self.common.policy.set_flat_weights(self.common.model_keeper.theta)

        # synchronize on updates completed
        for completion in update_completions:
            ray.get(completion)

        print('update workers complete')

        episodes_this_iteration = len(candidate_evals)
        self.episodes_so_far += episodes_this_iteration

        # Now sync the filters
        FilterManager.synchronize({
            DEFAULT_POLICY_ID: self.common.policy.observation_filter
        }, self._workers)

        def extract_columns(evaluations):
            rewards = np.fromiter((evaluation[1] for evaluation in evaluations), dtype=np.float32)
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
        # print('setstate ', self.common.policy.get_flat_weights().shape, state['weights'].shape)
        self.common.policy.set_flat_weights(state['weights'])
        self.common.policy.observation_filter = state["filter"]
        FilterManager.synchronize({
            DEFAULT_POLICY_ID: self.common.policy.observation_filter
        }, self._workers)
