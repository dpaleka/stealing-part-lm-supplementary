import numpy as np


def iterate_constraints(low, high, constraints, stopping_threshold=1e-5):
    """
    This is an iterative approximation of the Bellman-Ford algorithm.
    There is an additional parameter, stopping_threshold, which should be kept lower than the appropriately scaled tolerance.
    """

    if len(constraints) == 0:
        return {"low": low, "high": high}

    gap = np.mean(high - low)
    sampled_constraints = constraints
    while True:
        for highest_number, delta in sampled_constraints:
            difference = high[highest_number] + delta[highest_number] - delta
            difference[highest_number] = 1e9
            high = np.minimum(high, difference)

            max_other = low + delta - delta[highest_number]
            max_other[highest_number] = 0

            low[highest_number] = max(low[highest_number], np.max(max_other))
        if gap - np.mean(high - low) < stopping_threshold:
            break
        gap = np.mean(high - low)
    return {"low": low, "high": high}
