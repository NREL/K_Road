# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.

from __future__ import (
    absolute_import,
    division,
    print_function,
)

import numpy as np
from ray.rllib.utils import try_import_tf

tf1, tf, tfv = try_import_tf()


def compute_ranks(x):
    """Returns ranks in [0, len(x))

    Note: This is different from scipy.stats.rankdata, which returns ranks in
    [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= 0.5
    return y


def compute_wierstra_ranks(x):
    """
    section 3.1: http://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf
    as referenced in 2.1 of https://arxiv.org/pdf/1703.03864.pdf
    """
    y = x.size
    k = y - compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    u = np.maximum(0, np.log(y / 2 + 1) - np.log(k))
    u /= np.sum(u)
    u = u - 1 / y
    # max(0, log(k))
    # print('cwr: ', x.transpose(), u.transpose())
    return u


def make_session(single_threaded):
    if not single_threaded:
        return tf1.Session()
    return tf1.Session(
        config=tf1.ConfigProto(
            inter_op_parallelism_threads=1, intra_op_parallelism_threads=1))


def itergroups(items, group_size):
    """
    Yields blocks of the given size from iterating over items.
    Modifies the return value on each yield.
    """
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)


def batched_weighted_sum(weights, vecs, batch_size):
    """
    Computes the weighted (vector) sum of weights and vectors using a given block size
    """
    total = 0
    num_items_summed = 0
    for batch_weights, batch_vecs in zip(
            itergroups(weights, batch_size), itergroups(vecs, batch_size)):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        total += np.dot(
            np.asarray(batch_weights, dtype=np.float32),
            np.asarray(batch_vecs, dtype=np.float32))
        num_items_summed += len(batch_weights)
    return total, num_items_summed
