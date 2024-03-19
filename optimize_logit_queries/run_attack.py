"""
This file is the the basis for the logprob-free-attacks in "Stealing Part of a Production Language Model".
The data we report results on (the extracted OpenAI logits) is not released, 
but all the described algorithms + some ones that are only briefly described in the appendix are implemented here.
"""
# %%
import numpy as np

from bounders.bellman_ford import bellman_ford_bounder
from bounders.iterate_constraints import iterate_constraints
from bounders.floyd_warshall import floyd_warshall_bounder
from askers.simple import (
    normal_binary_search,
    simultaneous_binary_search,
    start_one_over_n_estimator,
    hyperrectangle_actual_center,
)
from askers.bias import bias_unexplored, bias_uncertain, query_mixed
from askers.distances import least_squares_all



def is_proper(real: np.ndarray) -> bool:
    """
    Let a vector `real` be *proper* if argmax(real) = 1 and 0 <= real <= 1.
    We run all our algorithms on proper vectors.
    """
    base_top_token = np.argmax(real)
    return (
        np.abs(real[base_top_token] - 1) < 1e-7
        and np.all(real >= 0)
        and np.all(real <= 1)
    )


def make_proper(real: np.ndarray, assumed_width: float | None) -> np.ndarray:
    """
    If scaling by 1/assumed_width, please report the tolerance in terms of the original scale.
    """
    if assumed_width is None:
        # Equivalent to real = (real - real.min()) / (real.max() - real.min())
        real = (real - real.max()) / (real.max() - real.min()) + 1
    else:
        # put max to 1, assume max - min <= assumed_width
        assert real.max() - real.min() <= assumed_width
        real = (real - real.max()) / assumed_width + 1

    assert is_proper(real)
    return real


def carlini_error(low, high, real):
    value = (high + low) / 2
    value += real[: len(value)].mean() - value.mean()
    return np.mean(np.square(value - real[: len(value)])) ** 0.5


def format_float(x) -> str:
    return f"{x:.3f}"


error_funcs = {
    "abs": lambda low, high, real: np.max(np.abs((high + low) / 2 - real)),
    "l2": lambda low, high, real: np.linalg.norm((high + low) / 2 - real),
    "bounds_abs": lambda low, high, _: np.max(np.abs(high - low)),
    "bounds_l2": lambda low, high, _: np.linalg.norm(high - low),
    "carlini_og_error": lambda low, high, real: carlini_error(low, high, real),
}

ERROR_TYPES = list(error_funcs.keys())

query_strategies = {
    "normal_binary_search": normal_binary_search,
    "simultaneous_binary_search": simultaneous_binary_search,
    "unexplored": bias_unexplored,
    "uncertain": bias_uncertain,
    "mixed": query_mixed,
    "least_squares_all": least_squares_all,
    "start_one_over_n": start_one_over_n_estimator,
    "hypercube_actual_center": hyperrectangle_actual_center,
}

bounds_computers = {
    "iterate_constraints": iterate_constraints,
    "bellman_ford": bellman_ford_bounder,
    "floyd_warshall": floyd_warshall_bounder,
}


def calculate_error(low, high, real):
    error = {name: func(low, high, real) for name, func in error_funcs.items()}
    return error


def run_attack(
    real: np.ndarray,
    tolerance: float,
    error_type: str,
    query_strategy: str,
    bounds_computer: str = "iterate_constraints",
    MAXQ: int = 1000,
    exit_on_success: bool = True,
    logging_freq: int | None = None,
    **kwargs,
) -> dict:
    assert (
        error_type in error_funcs
    ), f"Error type must be one of {list(error_funcs.keys())}"
    assert (
        query_strategy in query_strategies
    ), f"Query strategy must be one of {list(query_strategies.keys())}"
    assert (
        bounds_computer in bounds_computers
    ), f"Bounds computer must be one of {list(bounds_computers.keys())}"

    label = f"{query_strategy} {bounds_computer}"
    label += f' {kwargs["method"]}' if "method" in kwargs else ""
    label += f' alpha={format_float(kwargs["alpha"])}' if "alpha" in kwargs else ""
    label += f' beta={format_float(kwargs["beta"])}' if "beta" in kwargs else ""
    print(f"\nRunning {label}")

    query_strategy_func = query_strategies[query_strategy]
    bounds_computer_func = bounds_computers[bounds_computer]

    NTOK = len(real)
    low = np.zeros(NTOK)
    high = np.zeros(NTOK) + 1

    base_top_token = np.argmax(real)
    low[base_top_token] = high[base_top_token] = 1
    constraints = [(base_top_token, np.zeros(NTOK))]

    bounds_data = {
        "low": low,
        "high": high,
    }  # there can be additional fields for some bounds_computers, needed for some query_strategies

    current_error = calculate_error(low, high, real)[error_type]
    errors = {error_type: [] for error_type in ERROR_TYPES}

    steps = 0
    end_steps = None
    while steps < MAXQ:
        bias = query_strategy_func(
            low, high, constraints=constraints, bounds_data=bounds_data, **kwargs
        )
        top_token = np.argmax(real + bias)
        constraints.append((top_token, bias))

        bounds_data = bounds_computer_func(low, high, constraints)

        low, high = bounds_data["low"], bounds_data["high"]

        error_res = calculate_error(low, high, real)
        for error_type_iter in ERROR_TYPES:
            errors[error_type_iter].append(error_res[error_type_iter])
        current_error = error_res[error_type]

        if logging_freq is not None and steps % logging_freq == logging_freq // 2:
            print(f"Log gaps: {np.log(high - low)}")
            print(f"Error: {current_error}")

        steps += 1
        if current_error < tolerance and end_steps is None:
            end_steps = steps
            if exit_on_success:
                break

    return {
        "steps_to_tolerance": end_steps,
        "tolerance": tolerance,
        "error_type": error_type,
        "errors": errors,
        "label": label,
        "low": low,
        "guess": (low + high) / 2,
        "real": real,
    }


# %%
def run_paper_attacks(ground_truth_vector, tolerance, error_type):
    assert is_proper(ground_truth_vector)

    results = {}
    results["normal_binary_search"] = run_attack(
        real=ground_truth_vector,
        tolerance=tolerance,
        error_type=error_type,
        query_strategy="normal_binary_search",
        bounds_computer="iterate_constraints",
    )

    results["simultaneous_binary_search"] = run_attack(
        real=ground_truth_vector,
        tolerance=tolerance,
        error_type=error_type,
        query_strategy="simultaneous_binary_search",
        bounds_computer="iterate_constraints",
    )

    results["start_one_over_n"] = run_attack(
        real=ground_truth_vector,
        tolerance=tolerance,
        error_type=error_type,
        query_strategy="start_one_over_n",
        bounds_computer="iterate_constraints",
    )

    results["start_one_over_n_floyd"] = run_attack(
        real=ground_truth_vector,
        tolerance=tolerance,
        error_type=error_type,
        query_strategy="start_one_over_n",
        bounds_computer="floyd_warshall",
    )

    for alpha in [0.95]:
        results[f"iterate_constraints_bias_unexplored_{alpha}"] = run_attack(
            real=ground_truth_vector,
            tolerance=tolerance,
            error_type=error_type,
            query_strategy="unexplored",
            alpha=alpha,
            bounds_computer="iterate_constraints",
        )

    results["least_squares_all"] = run_attack(
        real=ground_truth_vector,
        tolerance=tolerance,
        error_type=error_type,
        query_strategy="least_squares_all",
        bounds_computer="floyd_warshall",
    )

    return {
        result_data["label"]: {
            "queries": result_data["steps_to_tolerance"],
            "guess": list(result_data["guess"]),
            "error_progression": {
                key: list(value) for key, value in result_data["errors"].items()
            },  # turn this off if you don't want it
        }
        for _, result_data in results.items()
    }


# %%
def test_on_random():
    tolerance = 1e-4
    error_type = "carlini_og_error"
    assumed_logit_max_minus_min = 35.0
    NTOK = 20
    # real = np.array([-15.4, 10.2, 10.1, -9.3, 0.4, 4.5])
    import json
    import random

    random.seed(10)
    np.random.seed(10)
    real = np.random.normal(0, 6, NTOK)
    real = make_proper(real, assumed_width=assumed_logit_max_minus_min)
    print(format_float(real.min()), format_float(real.max()))
    results = run_paper_attacks(real, tolerance, error_type)

    for key, value in results.items():
        print(key, value["queries"])
        print(json.dumps(value))


if __name__ == "__main__":
    test_on_random()
# test_on_random()
# %%
