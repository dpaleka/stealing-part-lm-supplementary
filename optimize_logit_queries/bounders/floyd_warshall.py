import numpy as np
from bounders.algos.floyd_warshall import floyd_warshall_fastest

"""
    Input
        adjacency_matrix[i,j] = distance to travel directly from node i to node j (without passing through other nodes)

        Notes:
        * If there is no edge connecting i->j then adjacency_matrix[i,j] should be equal to numpy.inf.
        * The diagonal of adjacency_matrix should be zero.

    Output
        An NxN NumPy array such that result[i,j] is the shortest distance to travel between node i and node j. If no such path exists then result[i,j] == numpy.inf
"""


def make_adjacency_matrix(edges, n_vertices):
    adj = np.full((n_vertices, n_vertices), np.inf)
    for u, v, w in edges:
        adj[u, v] = w
    for u in range(n_vertices):
        adj[u, u] = 0
    return adj


def floyd_warshall_bounder(low, high, constraints):
    # we compute from constraints
    NTOK = len(low)
    base_top_token, _ = constraints[0]

    # Create graph for max distance
    # Source will be an extra vertex NTOK
    list(range(NTOK + 1))
    edges = []
    for token, bias in constraints:
        for i in range(NTOK):
            if i != token:
                #     x[t] - x[i] >= bias[i] - bias[t]
                #     x[i] <= x[t] - (bias[i] - bias[t])
                a = bias[i] - bias[token]
                edges.append((token, i, -a))
    # x[s] = 0
    s = NTOK
    for i in range(NTOK):
        # x[i] <= x[s] + 1
        edges.append((s, i, 1))
        # x[s] <= x[i]
        edges.append((i, s, 0))
    # x[base_top_token] >= 1
    # x[s] <= x[base_top_token] - 1
    edges.append((base_top_token, s, -1))

    dist_max = floyd_warshall_fastest(make_adjacency_matrix(edges, NTOK + 1))
    # print(f"dist_max: {dist_max}")

    # Create graph for min distance
    # We're operating in negative space, so we'll flip the signs of the x's
    list(range(NTOK + 1))
    edges = []
    for token, bias in constraints:
        for i in range(NTOK):
            if i != token:
                #     x[t] - x[i] <= bias[i] - bias[t]
                #     -x[i] <= -x[t] + (bias[i] - bias[t])
                a = bias[i] - bias[token]
                edges.append((i, token, -a))
    # -x[s] = 0
    s = NTOK
    for i in range(NTOK):
        # -x[i] >= -1
        # -x[s] <= -x[i] + 1
        edges.append((i, s, 1))
        # -x[i] <= -x[s]
        edges.append((s, i, 0))
    # -x[base_top_token] <= -1
    # -x[base_top_token] <= -x[s] - 1
    edges.append((s, base_top_token, -1))

    dist_min = -floyd_warshall_fastest(make_adjacency_matrix(edges, NTOK + 1))
    # print(f"dist_min: {dist_min}")

    # new low and high are from the NTOK token
    eps = 1e-7
    new_low = dist_min[NTOK][0:NTOK]
    #    print(f"new_low: {new_low}")
    new_high = dist_max[NTOK][0:NTOK]
    #    print(f"new_high: {new_high}")
    assert np.all(new_low >= 0 - eps)
    assert np.all(new_high >= 0 - eps)
    assert np.all(new_low <= new_high + eps)
    low = np.maximum(low, new_low[0:NTOK])
    high = np.minimum(high, new_high[0:NTOK])
    return {"low": low, "high": high, "dist_min": dist_min, "dist_max": dist_max}
