import numpy as np
import random


def bias_unexplored(
    low,
    high,
    alpha: float = 0.5,
    constraints: list = None,
    method="exp_decay",
    **kwargs,
):
    """
    constraints: list of (token, bias) tuples. bias is a vector of length NTOK
    alpha: float, intended to be between 0 and 1, but depends on method

    returns: vector of length NTOK

    We want to incentivize tokens which haven't been the top token much yet.
    """
    # Frequency of each token being the top token
    NTOK = len(low)
    frequency = np.zeros(NTOK)
    for token, bias in constraints:
        frequency[token] += 1

    normalized_frequency = frequency / max(frequency.max(), 1)

    # How should we incentivize tokens which haven't been the top token much yet?
    # The default is to take 1 - (high + low) / 2; the average token should be approximately that
    # decay_bias should be a vector of length NTOK, average must be 0.5
    if method == "constant":
        decay_bias = np.zeros(NTOK) + 0.5
    elif method == "exp_decay":
        decay_bias = np.exp(-normalized_frequency / alpha)
        decay_bias /= decay_bias.mean() * 2
    elif method == "linear_decay":
        decay_bias = 1 - normalized_frequency * alpha
        decay_bias /= decay_bias.mean() * 2

    return 1 - low - decay_bias * (high - low)


def bias_uncertain(
    low, high, beta: float = 0.1, constraints: list = None, method="exp_decay", **kwargs
):
    """
    constraints: list of (token, bias) tuples. bias is a vector of length NTOK
    beta: float, intended to be between 0 and 1, but depends on method

    returns: vector of length NTOK

    We want to incentivize tokens for which (high - low) is large.
    """
    len(low)
    uncertainty = high - low
    normalized_uncertainty = uncertainty / uncertainty.max()

    if method == "exp_decay":
        decay_bias = np.exp(-normalized_uncertainty / beta)
        decay_bias /= decay_bias.mean() * 2

    return 1 - low - decay_bias * (high - low)


def query_mixed(low, high, callables: list, weights: list[float], **kwargs):
    # sample a method from callables with probability proportional to its weight
    callable = random.choices(callables, weights=weights)[0]
    return callable(low, high, **kwargs)
