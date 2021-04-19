import statistics
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


class ESTopNCoordinatedOptimizer(CoordinatedOptimizer):
    
    def __init__(
            self,
            theta,
            weights=None,
            candidates_per_iteration: int = 100,  # aka lambda
            num_survivors: int = 1,
            noise_stdev: float = 13,  #
            # noise_stdev: float = 65.1,  # Canonical ESRL Paper (.05 for 1.7M params) = .05*sqrt(1.7e6) = 65.1
            # noise_stdev:float = 3.84e-5 #(Canonical ESRL Paper (.05 for 1.7M params))
            # noise_stdev: float = 1.92e-4,
            # noise_stdev: float = 2 * 1.92e-4,
            # noise_stdev: float = 1.92e-4 * np.sqrt(10) / 1.5,
            # noise_stdev: float = .05,
            # step_scale:float = 1.0,
            step_scale: float = 1.0,
            noise_size: int = 67108864,
            random_seed: int = None,
            noise_table: Optional[NoiseTable] = None,
            **kwargs,
            ) -> None:
        self._theta = np.array(theta, dtype=np.float32)  # theta size: 67586
        
        self._candidates_per_iteration: int = candidates_per_iteration
        self.num_survivors: int = num_survivors
        self.step_scale: float = step_scale
        # self.noise_stdev: float = noise_stdev
        self.noise_stdev: float = noise_stdev / np.sqrt(self._theta.size)
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
        evaluations = sorted(evaluations, key=lambda evaluation: evaluation[0], reverse=True)
        survivors = evaluations[0:self.num_survivors]
        
        self._previous_theta = self._theta
        proposed_theta = np.empty(self._theta.size)
        for i in range(self.num_dimensions):
            proposed_theta[i] = statistics.mean((evaluation[1][i] for evaluation in survivors))
        
        if self.step_scale == 1.0:
            self._theta = proposed_theta
        else:
            delta = (proposed_theta - self._previous_theta) * self.step_scale
            self._theta = self._previous_theta + delta
    
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
