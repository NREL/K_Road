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
# from trainer.utils import (
#     batched_weighted_sum,
#     compute_wierstra_ranks,
#     )
from trainer.utils import (
    batched_weighted_sum,
    compute_wierstra_ranks,
    )


class ESRLCoordinatedOptimizer(CoordinatedOptimizer):
    
    def __init__(
            self,
            theta,
            alpha: float = .2,
            theta_decay: float = .001,
            noise_stdev:float = .05,
            # noise_stdev:float = 3.85e-3,
            candidates_per_iteration: int = 144,
            noise_size: int = 16777216,
            random_seed: int = None,
            noise_table: Optional[NoiseTable] = None,
            **kwargs,
            ) -> None:
        # print('ESRLCoordinatedOptimizer: ', candidates_per_iteration, alpha, theta_decay, noise_stdev, noise_size, random_seed)
        self.theta = theta
        self.alpha: float = alpha
        self.theta_decay: float = theta_decay
        self.noise_stdev: float = noise_stdev
        self.candidates_per_iteration: int = candidates_per_iteration
        self.noise_table: NoiseTable = NoiseTable(noise_size, random_seed) if noise_table is None else noise_table
        self.g = np.zeros(self.theta.size)
    
    def ask(self, num: Optional[int] = None) -> [any]:
        num = self.candidates_per_iteration if num is None else num
        num_pairs = math.ceil(num / 2)
        candidates: [any] = []
        for _ in range(num_pairs):
            offset = randint(0, sys.maxsize)
            candidates.append((offset, 1))
            candidates.append((offset, -1))
        return candidates
    
    def expand(self, candidate_message: any) -> any:
        noise_index, multiplier = candidate_message
        perturbation = multiplier * self.noise_stdev * self.noise_table.get(noise_index, len(self.theta))
        return self.theta + perturbation
    
    def tell_messages(self, evaluations: Iterable[Tuple[float, any]]) -> None:
        evaluations = evaluations if isinstance(evaluations, list) else list(evaluations)
        num_evaluations = len(evaluations)
        num_pairs = math.floor(len(evaluations) / 2)
        shape = (num_pairs, 2)
        scores = np.fromiter((evaluation[0] for evaluation in evaluations), dtype=np.float32).reshape(shape)
        ranks = compute_wierstra_ranks(scores)
        
        noise_indices = (evaluations[2 * i][1][0] for i in range(num_pairs))
        g, count = batched_weighted_sum(
            ranks[:, 0] - ranks[:, 1],
            (self.noise_table.get(index, len(self.theta)) for index in noise_indices),
            batch_size=500)
        
        g *= self.alpha / (num_evaluations * self.noise_stdev)
        self.g = g
        # Compute the new weights theta.
        self.theta = self.theta * (1.0 - self.theta_decay) + g
    
    def best(self) -> any:
        return 0, 0
    
    def status(self) -> dict:
        return {
            'weights_norm': np.sqrt(np.square(self.theta).sum()) / self.theta.size,
            'grad_norm':    np.sqrt(np.square(self.g).sum()) / self.g.size,
            }
