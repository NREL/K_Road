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


class ESEVRLCoordinatedOptimizer(CoordinatedOptimizer):
    
    # expected-value ES
    def __init__(
            self,
            theta,
            noise_stdev: float = 1.92e-4,  # 3.84e-5 (Canonical ESRL Paper (.05 for 1.7M params))
            candidates_per_iteration: int = 100,  # aka lambda
            noise_size: int = 67108864,
            random_seed: int = None,
            noise_table: Optional[NoiseTable] = None,
            **kwargs,
            ) -> None:
        self._theta = np.array(theta, dtype=np.float32)
        self.noise_stdev: float = noise_stdev * np.sqrt(self._theta.size)
        
        self._candidates_per_iteration: int = candidates_per_iteration
        self.noise_table: NoiseTable = NoiseTable(noise_size, random_seed) if noise_table is None else noise_table
        
        self._previous_theta = np.copy(self._theta)
    
    def ask(self, num: Optional[int] = None) -> [any]:
        num = self._candidates_per_iteration if num is None else num
        return [randint(0, sys.maxsize) for _ in range(num)]
    
    def expand(self, candidate_message: any) -> any:
        if candidate_message == 0:
            return self._theta
        
        return self._theta + self.noise_stdev * self.noise_table.get(candidate_message, self.num_dimensions)
    
    def tell(self, evaluations: Iterable[Tuple[float, any]]) -> None:
        # evaluations = sorted(evaluations, key=lambda evaluation: evaluation[0], reverse=True)
        evaluations = list(evaluations)
        distribution = np.fromiter((evaluation[0] for evaluation in evaluations), dtype=np.float32)
        minimum = np.min(distribution)
        maximum = np.max(distribution)
        print(minimum, maximum)
        normalized_distribution = (distribution - minimum) / (maximum - minimum + 1e-6)
        normalized_distribution /= np.sum(normalized_distribution)
        print(normalized_distribution)
        
        new_theta = self._previous_theta
        self._previous_theta = self._theta
        self._theta = new_theta
        for i in range(self.num_dimensions):
            new_theta[i] = sum((evaluation[1][i] * normalized_distribution[j]
                                for j, evaluation in enumerate(evaluations)))
    
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
