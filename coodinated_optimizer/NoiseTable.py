import numpy as np


class NoiseTable(object):
    
    def __init__(self, count, seed):
        self.noise = np.random.RandomState(seed).randn(count).astype(np.float32)
    
    def get(self, i, dim):
        assert dim <= len(self.noise)
        offset = i % (len(self.noise) - dim)
        return self.noise[offset:offset + dim]
    
    def sample_index(self, dim):
        return np.random.randint(0, len(self.noise) - dim + 1)
