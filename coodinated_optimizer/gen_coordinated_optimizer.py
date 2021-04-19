import sys
from random import randint
from typing import (
    Iterable,
    Optional,
    Tuple,
    )

import numpy as np

from coodinated_optimizer.NoiseTable import NoiseTable
from coodinated_optimizer.coordinated_optimizer import CoordinatedOptimizer


class Candidate:
    
    def __init__(self, id, parent, theta, mu, sigma):
        self.id = id
        self.parent = parent
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.children = []

"""
+ Heuristic search method:
    + Best search locations are:
        + Between good candidates:
            + closer to better candidate
            + closer to higher variance candidate
        + Between a candidate and the 'background'
            + somewhere where variance and mean are high
        + Places where both mean and variance are high
            + formula?
    + Estimating a candidate's value:
        + if there is no sensor noise, then the evaluation is the estimate is the true value
        + with sensor noise, want to inspect relative's evaluations as well, with decreasing weight based on distance
    + Choosing candidates to combine:
        + could combine 0 or more, with the 'background' being a virtual parent as well (this is akin to mutation)
    

    
"""

class GenCoordinatedOptimizer(CoordinatedOptimizer):
    
    def __init__(
            self,
            theta,
            mu: float = 0.0,
            sigma: float = .5,
            # selection_proportion: float = .10,
            candidates_per_iteration: int = 100,
            noise_size: int = 67108864,
            random_seed: int = None,
            noise_table: Optional[NoiseTable] = None,
            **kwargs,
            ) -> None:
        
        self.population = {0: Candidate(0, None, theta, mu, sigma)}
        
        self._candidates_per_iteration: int = candidates_per_iteration
        self.noise_table: NoiseTable = NoiseTable(noise_size, random_seed) if noise_table is None else noise_table
    
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
        # evaluations = list(evaluations)
        
        population = self.population
        
        for evaluation in evaluations:
            population.append(evaluation)
        
        # evaluations = sorted(evaluations, key=lambda evaluation: evaluation[0], reverse=True)
        
        # selection_size = math.ceil(self._selection_proportion * len(evaluations))
        # del evaluations[selection_size:]
        
        # for i in range(self.num_dimensions):
        #     values = np.fromiter((evaluation[1][i] for evaluation in evaluations), dtype=np.float32)
        #
        #     mean = np.average(values, weights=self._weights)
        #     self._mu[i] = mean
        #     self._sigma[i] = math.sqrt(np.average((values - mean) ** 2, weights=self._weights))
        #
        #     # self._mu[i] = np.mean(values, weights=self._weights)
        #     # self._sigma[i] = np.std(values, weights=self._weights)
    
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
