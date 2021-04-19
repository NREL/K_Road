import math
import sys
from random import (
    gauss,
    randint,
    )
from typing import (
    Iterable,
    Optional,
    Tuple,
    )

import numpy as np

from coodinated_optimizer.NoiseTable import NoiseTable
from coodinated_optimizer.coordinated_optimizer import CoordinatedOptimizer


class ESBestSpreadCoordinatedOptimizer(CoordinatedOptimizer):
    
    def __init__(
            self,
            theta,
            weights=None,
            candidates_per_iteration: int = 100,  # aka lambda
            noise_stdev: float = 1000,
            # noise_stdev: float = 13,  #
            # noise_stdev: float = 65.1,  # Canonical ESRL Paper (.05 for 1.7M params) = .05*sqrt(1.7e6) = 65.1
            # noise_stdev:float = 3.84e-5 #(Canonical ESRL Paper (.05 for 1.7M params))
            # noise_stdev: float = 1.92e-4,
            # noise_stdev: float = 2 * 1.92e-4,
            # noise_stdev: float = 1.92e-4 * np.sqrt(10) / 1.5,
            # noise_stdev: float = .05,
            # step_scale:float = 1.0,
            noise_size: int = 67108864,
            random_seed: int = None,
            noise_table: Optional[NoiseTable] = None,
            **kwargs,
            ) -> None:
        self._theta = np.array(theta, dtype=np.float32)  # theta size: 67586
        
        self._candidates_per_iteration: int = candidates_per_iteration
        self.noise_stdev: float = noise_stdev
        self.noise_table: NoiseTable = NoiseTable(noise_size, random_seed) if noise_table is None else noise_table
        self._previous_theta = np.copy(self._theta)
        self.best_evaluation = None
        print('ESBestSpreadCoordinatedOptimizer::init', self._theta.size, noise_stdev, self.noise_stdev,
              candidates_per_iteration)
    
    def ask(self, num: Optional[int] = None) -> [any]:
        num = self._candidates_per_iteration if num is None else num
        return [(randint(0, sys.maxsize), gauss(0.0, self.noise_stdev)) for _ in range(num)]
    
    def expand(self, candidate_message: any) -> any:
        if candidate_message[0] == 0 or candidate_message[1] == 0.0:
            return self._theta
        
        delta = candidate_message[1] / math.sqrt(self.num_dimensions)
        # print('exp ', delta, candidate_message[1])
        return self._theta + delta * self.noise_table.get(candidate_message[0], self.num_dimensions)
    
    def tell(self, evaluations: Iterable[Tuple[float, any]]) -> None:
        def key(evaluation):
            return evaluation[0]
        
        # best_evaluation = max(evaluations, key=key)
        be = max(evaluations, key=key)
        best_evaluation = be
        if self.best_evaluation is not None:
            best_evaluation = max((self.best_evaluation, best_evaluation), key=key)
        # print('best evaluation',
        #       best_evaluation[0],
        #       be[0],
        #       'none' if self.best_evaluation is None else self.best_evaluation[0],
        #       best_evaluation[1][0],
        #       'none' if self.best_evaluation is None else self.best_evaluation[1][0])
        self.best_evaluation = best_evaluation
        self._previous_theta = self._theta
        self._theta = self.best_evaluation[1]
    
    def best(self) -> any:
        return 0, 0.0
    
    def status(self) -> dict:
        return {
            'weights_norm': np.sqrt(np.square(self._theta).sum()) / self.num_dimensions,
            'grad_norm':    np.sqrt(np.square(self._theta - self._previous_theta).sum()) / self.num_dimensions,
            }
    
    @property
    def num_dimensions(self) -> int:
        return self._theta.size
