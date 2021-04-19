import math
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


class CanonicalESCoordinatedOptimizer(CoordinatedOptimizer):
    
    def __init__(
            self,
            theta,
            # sigma: float = .05,  # mutation step-size
            sigma: float = .435,  # mutation step-size (8.7X single-point norm)
            # noise_stdev: float = 1.92e-4,
            candidates_per_iteration: int = 144,  # aka lambda
            noise_size: int = 67108864,
            random_seed: int = None,
            noise_table: Optional[NoiseTable] = None,
            **kwargs,
            ) -> None:
        print('theta size:', theta.size)
        self._theta = np.array(theta, dtype=np.float32)  # theta size: 67586
        self._sigma: float = sigma
        
        self._candidates_per_iteration: int = candidates_per_iteration
        self.noise_table: NoiseTable = NoiseTable(noise_size, random_seed) if noise_table is None else noise_table
        
        self._weights = np.fromiter(
            (math.log(candidates_per_iteration + 1.5) - math.log(i + 1) for i in range(candidates_per_iteration)),
            dtype=np.float32)
        
        self._weights /= np.sum(self._weights)
        self._previous_theta = np.copy(self._theta)
    
    def ask(self, num: Optional[int] = None) -> [any]:
        num = self._candidates_per_iteration if num is None else num
        return [randint(0, sys.maxsize) for _ in range(num)]
    
    def expand(self, candidate_message: any) -> any:
        if candidate_message == 0:
            return self._theta
        
        return self._theta + self._sigma * self.noise_table.get(candidate_message, self.num_dimensions)
    
    def tell(self, evaluations: Iterable[Tuple[float, any]]) -> None:
        evaluations = sorted(evaluations, key=lambda evaluation: evaluation[0], reverse=True)
        
        self._previous_theta = self._theta
        proposed_theta = np.empty(self._theta.size)
        for i in range(self.num_dimensions):
            proposed_theta[i] = sum((evaluation[1][i] * self._weights[rank]
                                     for rank, evaluation in enumerate(evaluations)))
        self._theta = proposed_theta
    
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
