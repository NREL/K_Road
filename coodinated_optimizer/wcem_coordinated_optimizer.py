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


class WCEMCoordinatedOptimizer(CoordinatedOptimizer):
    
    def __init__(
            self,
            mu,
            sigma: float = .25,
            # selection_proportion: float = .10,
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
        
        # self._selection_proportion: float = selection_proportion
        self._sigma = np.zeros(mu.size, dtype=np.float32) + sigma
        self._best_candidate: Optional[Tuple[float, any]] = None
        self._remember_best: bool = remember_best
        
        self._weights = np.fromiter(
            (math.log(candidates_per_iteration + 1.5) - math.log(i + 1) for i in range(candidates_per_iteration)),
            dtype=np.float32)
        
        # self._weights = np.fromiter(
        #     (math.exp(-.2*i)for i in range(candidates_per_iteration)),
        #     dtype=np.float32)
        
        self._weights /= np.sum(self._weights)
        pprint(self._weights)
    
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
        
        # selection_size = math.ceil(self._selection_proportion * len(evaluations))
        # del evaluations[selection_size:]
        
        for i in range(self.num_dimensions):
            values = np.fromiter((evaluation[1][i] for evaluation in evaluations), dtype=np.float32)
            
            mean = np.average(values, weights=self._weights)
            self._mu[i] = mean
            self._sigma[i] = math.sqrt(np.average((values - mean) ** 2, weights=self._weights))
            
            # self._mu[i] = np.mean(values, weights=self._weights)
            # self._sigma[i] = np.std(values, weights=self._weights)
    
    def best(self) -> any:
        return 0
    
    def status(self) -> dict:
        return {
            'weights_norm': np.square(self._mu).sum(),
            'stdev_norm':   np.square(self._sigma).sum(),
            }
    
    @property
    def num_dimensions(self) -> int:
        return self._mu.size
