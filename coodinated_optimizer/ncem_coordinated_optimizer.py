import math
import sys
from pprint import pprint
from random import randint
from typing import (
    Iterable,
    Optional,
    Tuple,
    )

import numpy as np

from coodinated_optimizer.NoiseTable import NoiseTable
from coodinated_optimizer.coordinated_optimizer import CoordinatedOptimizer


class NCEMCoordinatedOptimizer(CoordinatedOptimizer):
    
    def __init__(
            self,
            mu,
            sigma: float = .25,
            selection_proportion: float = .10,
            candidates_per_iteration: int = 144,
            noise_size: int = 67108864,
            random_seed: int = None,
            noise_table: Optional[NoiseTable] = None,
            remember_best: bool = False,
            **kwargs,
            ) -> None:
        self._mu = np.array(mu, dtype=np.float32)
        
        self._candidates_per_iteration: int = candidates_per_iteration
        self.noise_table: NoiseTable = NoiseTable(noise_size, random_seed) if noise_table is None else noise_table
        
        self._selection_proportion: float = selection_proportion
        self._selection_size: int = math.ceil(self._selection_proportion * candidates_per_iteration)
        self._sigma = np.zeros(mu.size, dtype=np.float32) + sigma
        self._best_candidate: Optional[Tuple[float, any]] = None
        self._remember_best: bool = remember_best
        
        self._weights = np.fromiter((1.0 for i in range(self._selection_size)), dtype=np.float32)
        
        # self._weights = np.fromiter(
        #     ((1.0 if i < selection_proportion * candidates_per_iteration else 0.0) for i in
        #      range(candidates_per_iteration)), dtype=np.float32)
        
        # self._weights = np.fromiter(
        #     (math.log(candidates_per_iteration + 1.5) - math.log(i + 1) for i in range(candidates_per_iteration)),
        #     dtype=np.float32)
        
        # self._weights = np.fromiter(
        #     (math.exp(-.2*i)for i in range(candidates_per_iteration)),
        #     dtype=np.float32)
        
        # self._weights /= np.sum(self._weights)
        # pprint(self._weights)
    
    def ask(self, num: Optional[int] = None) -> [any]:
        num = self._candidates_per_iteration if num is None else num
        return [randint(0, sys.maxsize) for _ in range(num)]
    
    def expand(self, candidate_message: any) -> any:
        if candidate_message == 0:
            if self._best_candidate is None:
                return self._mu
            else:
                return self._best_candidate[1]
        
        return self._mu + self._sigma * self.noise_table.get(candidate_message, self.num_dimensions)
    
    def tell(self, evaluations: Iterable[Tuple[float, any]]) -> None:
        evaluations = sorted(evaluations, key=lambda evaluation: evaluation[0], reverse=True)
        
        if self._remember_best and (self._best_candidate is None or evaluations[0][0] > self._best_candidate[0]):
            self._best_candidate = evaluations[0]
        
        del evaluations[self._selection_size:]
        
        # weight evaluations based on their likelihood
        print(self.num_dimensions, len(evaluations))
        weights = np.empty(len(evaluations), dtype=np.float32)
        for j, evaluation in enumerate(evaluations):
            relative_log_likelihood = 0.0
            candidate = evaluation[1]
            
            relative_log_likelihood = np.sum(np.square((candidate - self._mu) / self._sigma))
            relative_log_likelihood = -relative_log_likelihood
            
            # for i in range(self.num_dimensions):
            #     sigma = self._sigma[i]
            #     mu = self._mu[i]
            #     x = candidate[i]
            #     this_relative_log_likelihood = - (((x - mu) / sigma) ** 2)
            #     relative_log_likelihood += this_relative_log_likelihood
            #     # print(j, i, sigma, mu, x, this_relative_log_likelihood, relative_log_likelihood)
            weights[j] = relative_log_likelihood
            # realtive_likelihood = math.exp(relative_log_likelihood)
            # print('rl:', realtive_likelihood, relative_log_likelihood)
            # weights[j] = self._weights[j] / (realtive_likelihood + 1e-6)
        
        print('w1')
        pprint(weights)
        weights += -460 - np.min(weights)
        print('w2')
        pprint(weights)
        weights = np.exp(weights)

        print('w2 1')
        pprint(weights)
        weights = 1.0 / (weights + 1e-200)
        print('w2 2')
        pprint(weights)
        weights /= np.max(weights)
        print('w3')
        pprint(weights)
        weights *= self._weights
        print('w4')
        pprint(weights)
        
        for i in range(self.num_dimensions):
            values = np.fromiter((evaluation[1][i] for evaluation in evaluations), dtype=np.float32)
            
            mean = np.average(values, weights=weights)
            self._mu[i] = mean
            self._sigma[i] = math.sqrt(np.average((values - mean) ** 2, weights=weights))
            
            # self._mu[i] = np.mean(values, weights=self._weights)
            # self._sigma[i] = np.std(values, weights=self._weights)
    
    def best(self) -> any:
        return 0
    
    def status(self) -> dict:
        return {
            'weights_norm': np.sqrt(np.square(self._mu).sum()) / self.num_dimensions,
            'stdev_norm':   np.sqrt(np.square(self._sigma).sum()) / self.num_dimensions,
            }
    
    @property
    def num_dimensions(self) -> int:
        return self._mu.size
