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


class ESWeightedRLCoordinatedOptimizer(CoordinatedOptimizer):
    
    def __init__(
            self,
            theta,
            weights=None,
            candidates_per_iteration: int = 100,  # aka lambda
            noise_stdev: float = .05,
            noise_size: int = 67108864,
            random_seed: int = None,
            noise_table: Optional[NoiseTable] = None,
            **kwargs,
            ) -> None:
        self._theta = np.array(theta, dtype=np.float32)
        
        self._candidates_per_iteration: int = candidates_per_iteration
        self.noise_stdev: float = noise_stdev
        self.noise_table: NoiseTable = NoiseTable(noise_size, random_seed) if noise_table is None else noise_table
        
        # self._weights = np.fromiter(
        #     (2 * (1 - x) - 1
        #      for x in ((i / (candidates_per_iteration - 1))
        #                for i in range(candidates_per_iteration))),
        #     dtype=np.float32)
        
        # self._weights = np.fromiter(
        #     ((1 - x)
        #      for x in ((i / (candidates_per_iteration - 1))
        #                for i in range(candidates_per_iteration))),
        #     dtype=np.float32)
        
        # self._weights = np.zeros(candidates_per_iteration, dtype=np.float32)
        # self._weights[0] = 1.0
        
        # self._weights = np.fromiter(
        #     (1.0 if x <= .02 else 0
        #      for x in ((i / (candidates_per_iteration - 1))
        #                for i in range(candidates_per_iteration))),
        #     dtype=np.float32)
        
        self._weights = weights
        # self._weights /= np.sum(np.abs(self._weights))
        
        self._previous_theta = np.copy(self._theta)
    
    def ask(self, num: Optional[int] = None) -> [any]:
        num = self._candidates_per_iteration if num is None else num
        return [randint(0, sys.maxsize) for _ in range(num)]
    
    def expand(self, candidate_message: any) -> any:
        if candidate_message == 0:
            return self._theta
        
        return self._theta + self.noise_stdev * self.noise_table.get(candidate_message, self.num_dimensions)
    
    def tell(self, evaluations: Iterable[Tuple[float, any]]) -> None:
        evaluations = sorted(evaluations, key=lambda evaluation: evaluation[0], reverse=True)
        
        new_theta = self._previous_theta
        self._previous_theta = self._theta
        self._theta = new_theta
        for i in range(self.num_dimensions):
            new_theta[i] = sum((evaluation[1][i] * self._weights[rank]
                                for rank, evaluation in enumerate(evaluations)))
        
        # print('del:', np.sum(np.square(new_theta - evaluations[0][1])))
    
    def best(self) -> any:
        return 0
    
    def status(self) -> dict:
        return {
            'weights_norm': np.sqrt(np.square(self._theta).sum()) / self.num_dimensions,
            'grad_norm':    np.sqrt(np.square(self._theta - self._previous_theta).sum()) / self.num_dimensions,
            }
    
    @property
    def num_dimensions(self) -> int:
        return self._theta.size
