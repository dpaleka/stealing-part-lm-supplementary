import numpy as np

from askers.simple import simultaneous_binary_search


def least_squares_all(
    low,
    high,
    min_dist=None,
    max_dist=None,
    constraints=None,
    bounds_data=None,
    **kwargs,
):
    """
    Input:
        low: vector of length NTOK
        high: vector of length NTOK
        min_dist: matrix of shape (NTOK, NTOK).
                  min_dist[i, j] is a lower bound on x_j - x_i
        max_dist: matrix of shape (NTOK, NTOK)
                  max_dist[i, j] is an upper bound on x_j - x_i
                  It is true that min_dist[i, j] <= max_dist[i, j] and min_dist[i, j] = -max_dist[j, i]
        constraints: list of (token, bias) tuples. bias is a vector of length NTOK
    """

    if min_dist is None or max_dist is None:
        assert bounds_data is not None
        if "dist_min" not in bounds_data or "dist_max" not in bounds_data:
            return simultaneous_binary_search(low, high, **kwargs)
        else:
            min_dist = bounds_data["dist_min"]
            max_dist = bounds_data["dist_max"]

    base_top_token, _ = constraints[0]

    # We want each x_i + v_i to be the same
    # Hence for each pair (i, j), we want x_j - x_i to be centered around 0
    # So v_j - v_i should be -(max_dist[i, j] + min_dist[i, j]) / 2
    # We don't need to use low and high because one variable is fixed to 1

    # This system is overconstrained, so we use least squares

    NTOK = len(low)
    A = np.zeros((NTOK * NTOK + 1, NTOK))
    b = np.zeros((NTOK * NTOK + 1))
    for i in range(NTOK):
        for j in range(NTOK):
            if i == j:
                continue
            A[i * NTOK + j, i] = -1
            A[i * NTOK + j, j] = 1
            b[i * NTOK + j] = -(max_dist[i, j] + min_dist[i, j]) / 2

    # set v[base_top_token] = 1
    A[-1, base_top_token] = 1
    b[-1] = 1

    # We want to minimize ||Ax - b||^2
    v, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    assert residuals.item() - np.linalg.norm(A @ v - b) ** 2 < 1e-6

    return v
