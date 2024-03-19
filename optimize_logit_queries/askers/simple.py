# %%
import numpy as np
import torch


def normal_binary_search(low, high, **kwargs):
    bias = np.zeros(len(low))
    q = np.argmax(high - low)
    bias[q] = 1 - ((high + low) / 2)[q]
    return bias


def simultaneous_binary_search(low, high, **kwargs):
    # hyperrectangle relaxation
    return 1 - (high + low) / 2


def start_one_over_n_estimator(low, high, constraints=None, **kwargs):
    # set s.t. the base token interval has prob 1/n of going to the top
    # original interval is [low, high]
    base_top_token, _ = constraints[0]
    NTOK = len(low)
    c = np.exp(-np.log(NTOK) / (NTOK - 1))
    # c fraction of each of the other intervals is above 1
    # [low, high] -> [low + r, high + r] s.t. (1 - (low + r)) = c * (high - low)
    r = 1 - (1 - c) * low - c * high
    assert r[base_top_token] == 0
    return r


def hyperrectangle_actual_center(
    low, high, constraints=None, distribution="uniform", **kwargs
):
    """
    hypercube relaxation, but correctly
    we want f_i(v) = P[x_i + v_i >= max_{j != i} x_j + v_j] to be 1/n, for all i
    it is true that
    f_i = \int_{low[i]}^{high[i]} \prod_{j != i} P[x_j <= x_i + v_i - v_j] dx_i

    for uniform distribution, there is an exact integral for f_i, which we approximate by sampling
    similarly for normal distribution. however we don't have mu and sigma yet, so raise notimplementederror

    we actually do logf_i = \sum_{j != i} log P[x_j <= x_i + v_i - v_j]
    """

    raise NotImplementedError


# %%
