# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.

from __future__ import (
    absolute_import,
    division,
    print_function,
)

from ray.rllib.agents.es.es_tf_policy import ESTFPolicy, rollout
from ray.rllib.utils import try_import_tf

from trainer import utils

tf = try_import_tf()

from collections import namedtuple
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
# from ray.rllib.utils.memory import ray_get_and_free
from ray.rllib.utils import FilterManager

logger = logging.getLogger(__name__)

Result = namedtuple("Result", [
    "noise_indices", "noisy_returns", "noisy_lengths",
    "eval_returns", "eval_lengths"
])

# yapf: disable
# __sphinx_doc_begin__
DEFAULT_CONFIG = with_common_config({
    "theta_decay": 0.001,
    "alpha": .2,
    "noise_stdev": 0.02,
    "episodes_per_batch": 1000,
    "train_batch_size": 10000,
    "eval_prob": 0.003,
    # "return_proc_mode":   "centered_rank",
    "num_workers": 10,
    # "stepsize": 0.01,
    "observation_filter": "MeanStdFilter",
    # "observation_filter": None,
    "noise_size": 300000000,
    "report_length": 10,
})


# __sphinx_doc_end__
# yapf: enable


@ray.remote
def create_shared_noise(count):
    """Create a large array of noise to be shared by all workers."""
    seed = 123
    # seed = datetime.now()  # could be set to a fixed value for reproducible results
    noise = np.random.RandomState(seed).randn(count).astype(np.float32)
    return noise


class SharedNoiseTable(object):

    def __init__(self, noise):
        self.noise = noise
        assert self.noise.dtype == np.float32

    def get(self, i, dim):
        return self.noise[i:i + dim]

    def sample_index(self, dim):
        return np.random.randint(0, len(self.noise) - dim + 1)


@ray.remote
class Worker(object):

    def __init__(self,
                 config,
                 policy_params,
                 env_creator,
                 noise,
                 min_task_runtime=0.2):
        self.min_task_runtime = min_task_runtime
        self.config = config
        self.policy_params = policy_params
        self.noise = SharedNoiseTable(noise)

        self.env = env_creator(config["env_config"])
        from ray.rllib import models

        self.preprocessor = models.ModelCatalog.get_preprocessor(
            self.env, config["model"])

        self.sess = utils.make_session(single_threaded=True)
        # self.policy = GenericPolicy(
        #     self.sess, self.env.action_space, self.env.observation_space,
        #     self.preprocessor, config["observation_filter"], config["model"],
        #     **policy_params)
        policy_cls = self.get_policy_class(config)
        self.policy = policy_cls(
            obs_space=self.env.observation_space,
            action_space=self.env.action_space,
            config=config)

    @staticmethod
    def get_policy_class(config):
        if config["framework"] == "torch":
            from ray.rllib.agents.es.es_torch_policy import ESTorchPolicy

            policy_cls = ESTorchPolicy
        else:
            policy_cls = ESTFPolicy
        return policy_cls

    @property
    def filters(self):
        return {DEFAULT_POLICY_ID: self.policy.observation_filter}

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

    def rollout(self, timestep_limit):
        rollout_rewards, rollout_length = rollout(
            self.policy,
            self.env,
            timestep_limit=timestep_limit,
            add_noise=False)
        return rollout_rewards, rollout_length

    def do_rollouts(self, params, timestep_limit=None):
        # Set the network weights.
        self.policy.set_weights(params)

        noise_indices, noisy_returns, noisy_lengths = [], [], []
        eval_returns, eval_lengths = [], []

        # Perform some rollouts with noise.
        task_tstart = time.time()
        while (len(noise_indices) == 0
               or time.time() - task_tstart < self.min_task_runtime):

            if np.random.uniform() < self.config["eval_prob"]:
                # Do an evaluation run with no perturbation.
                self.policy.set_weights(params)
                rewards, length = self.rollout(timestep_limit)
                eval_returns.append(rewards.sum())
                eval_lengths.append(length)
            else:
                # Do a regular run with parameter perturbations.
                noise_index = self.noise.sample_index(self.policy.num_params)

                perturbation = self.config["noise_stdev"] * self.noise.get(
                    noise_index, self.policy.num_params)

                # These two sampling steps could be done in parallel on
                # different actors letting us update twice as frequently.
                self.policy.set_weights(params + perturbation)
                rewards_pos, lengths_pos = self.rollout(timestep_limit)

                self.policy.set_weights(params - perturbation)
                rewards_neg, lengths_neg = self.rollout(timestep_limit)

                noise_indices.append(noise_index)
                noisy_returns.append(
                    [rewards_pos.sum(),
                     rewards_neg.sum()])
                noisy_lengths.append([lengths_pos, lengths_neg])

        return Result(
            noise_indices=noise_indices,
            noisy_returns=noisy_returns,
            noisy_lengths=noisy_lengths,
            eval_returns=eval_returns,
            eval_lengths=eval_lengths)


class ESActualTrainer(Trainer):
    """Large-scale implementation of Evolution Strategies in Ray."""

    _name = "ES"
    _default_config = DEFAULT_CONFIG

    @override(Trainer)
    def _init(self, config, env_creator):
        policy_params = {"action_noise_std": 0.0}

        self.alpha = config['alpha']
        self.theta_decay = config['theta_decay']

        env = env_creator(config["env_config"])
        from ray.rllib import models

        preprocessor = models.ModelCatalog.get_preprocessor(env)

        self.sess = utils.make_session(single_threaded=False)
        self.policy = policies.GenericPolicy(
            self.sess, env.action_space, env.observation_space, preprocessor,
            config["observation_filter"], config["model"], **policy_params)
        # self.optimizer = optimizers.Adam(self.policy, config["stepsize"])
        self.report_length = config["report_length"]

        # Create the shared noise table.
        logger.info("Creating shared noise table.")
        noise_id = create_shared_noise.remote(config["noise_size"])
        self.noise = SharedNoiseTable(ray.get(noise_id))

        # Create the actors.
        logger.info("Creating actors.")
        self._workers = [
            Worker.remote(config, policy_params, env_creator, noise_id)
            for _ in range(config["num_workers"])
        ]

        self.episodes_so_far = 0
        self.reward_list = []
        self.tstart = time.time()
        self.episode_reward_mean = None
        self.episode_len_mean = None

        print('ESActualTrainer init, logdir: ', self._logdir, self.logdir)

    @override(Trainer)
    def _train(self):
        config = self.config

        theta = self.policy.get_weights()
        assert theta.dtype == np.float32

        # Put the current policy weights in the object store.
        theta_id = ray.put(theta)
        # Use the actors to do rollouts, note that we pass in the ID of the
        # policy weights.
        results, num_episodes, num_timesteps = self._collect_results(
            theta_id, config["episodes_per_batch"], config["train_batch_size"])

        all_noise_indices = []
        all_training_returns = []
        all_training_lengths = []
        all_eval_returns = []
        all_eval_lengths = []

        # Loop over the results.
        for result in results:
            all_eval_returns += result.eval_returns
            all_eval_lengths += result.eval_lengths

            all_noise_indices += result.noise_indices
            all_training_returns += result.noisy_returns
            all_training_lengths += result.noisy_lengths

        assert len(all_eval_returns) == len(all_eval_lengths)
        assert (len(all_noise_indices) == len(all_training_returns) ==
                len(all_training_lengths))

        self.episodes_so_far += num_episodes

        # Assemble the results.
        eval_returns = np.array(all_eval_returns)
        eval_lengths = np.array(all_eval_lengths)
        noise_indices = np.array(all_noise_indices)
        noisy_returns = np.array(all_training_returns)
        noisy_lengths = np.array(all_training_lengths)

        # Process the returns.
        # if config["return_proc_mode"] == "centered_rank":
        #     proc_noisy_returns = utils.compute_centered_ranks(noisy_returns)
        # else:
        #     raise NotImplementedError(config["return_proc_mode"])

        # Compute and take a step.
        # print('nrs: ', noisy_returns.shape, noisy_returns)

        # proc_noisy_returns = utils.compute_centered_ranks(noisy_returns)
        proc_noisy_returns = utils.compute_wierstra_ranks(noisy_returns)
        g, count = utils.batched_weighted_sum(
            proc_noisy_returns[:, 0] - proc_noisy_returns[:, 1],
            (self.noise.get(index, self.policy.num_params) for index in noise_indices),
            batch_size=500)
        g *= self.alpha / (noisy_returns.size * self.config["noise_stdev"])

        assert (g.shape == (self.policy.num_params,) and g.dtype == np.float32
                and count == len(noise_indices))
        # Compute the new weights theta.
        theta = theta * (1.0 - self.theta_decay) + g
        # theta, update_ratio = self.optimizer.update(-g +
        #                                             config["l2_coeff"] * theta)
        # Set the new weights in the local copy of the policy.
        self.policy.set_weights(theta)
        # # Store the rewards
        # if len(all_eval_returns) > 0:
        #     self.reward_list.append(np.mean(eval_returns))

        # Now sync the filters
        FilterManager.synchronize({
            DEFAULT_POLICY_ID: self.policy.get_filter()
        }, self._workers)

        info = {
            "weights_norm": np.square(theta).sum(),
            "grad_norm": np.square(g).sum(),
            # "update_ratio":       update_ratio,
            "episodes_this_iter": 2 * noisy_lengths.size,
            "episodes_so_far": self.episodes_so_far,
        }

        reward_mean = np.mean(self.reward_list[-self.report_length:])

        if eval_returns.size > 0:
            self.episode_reward_mean = eval_returns.mean()
            self.episode_len_mean = eval_lengths.mean()

        result = dict(
            episode_reward_mean=self.episode_reward_mean,
            episode_len_mean=self.episode_len_mean,
            timesteps_this_iter=noisy_lengths.sum(),
            info=info)

        return result

    @override(Trainer)
    def compute_action(self, observation):
        return self.policy.compute(observation, update=False)[0]

    @override(Trainer)
    def _stop(self):
        # workaround for https://github.com/ray-project/ray/issues/1516
        for w in self._workers:
            w.__ray_terminate__.remote()

    def _collect_results(self, theta_id, min_episodes, min_timesteps):
        num_episodes, num_timesteps = 0, 0
        results = []
        while num_episodes < min_episodes or num_timesteps < min_timesteps:
            logger.info(
                "Collected {} episodes {} timesteps so far this iter".format(
                    num_episodes, num_timesteps))
            rollout_ids = [
                worker.do_rollouts.remote(theta_id) for worker in self._workers
            ]
            # Get the results of the rollouts.
            for result in ray_get_and_free(rollout_ids):
                results.append(result)
                # Update the number of episodes and the number of timesteps
                # keeping in mind that result.noisy_lengths is a list of lists,
                # where the inner lists have length 2.
                # num_episodes += sum(len(pair) for pair in result.noisy_lengths)
                num_episodes += sum(len(pair) for pair in result.noisy_lengths)
                num_timesteps += sum(
                    sum(pair) for pair in result.noisy_lengths)
                # num_episodes += len(result.noisy_lengths)
                # num_timesteps += sum(result.noisy_lengths)

        return results, num_episodes, num_timesteps

    def __getstate__(self):
        return {
            "weights": self.policy.get_weights(),
            "filter": self.policy.get_filter(),
            "episodes_so_far": self.episodes_so_far,
        }

    def __setstate__(self, state):
        self.episodes_so_far = state["episodes_so_far"]
        self.policy.set_weights(state["weights"])
        self.policy.set_filter(state["filter"])
        FilterManager.synchronize({
            DEFAULT_POLICY_ID: self.policy.get_filter()
        }, self._workers)
