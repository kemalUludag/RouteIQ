import numpy as np

from src.routing import (
    calculate_routes_cost
)


def calculate_matrix_mae(
    predicted_matrix,
    true_matrix
) -> float:

    predicted_values = []
    true_values = []

    num_nodes = len(
        true_matrix
    )

    for i in range(num_nodes):
        for j in range(num_nodes):

            if i == j:
                continue

            predicted_values.append(
                predicted_matrix[i][j]
            )

            true_values.append(
                true_matrix[i][j]
            )

    predicted_array = np.asarray(
        predicted_values,
        dtype=float
    )

    true_array = np.asarray(
        true_values,
        dtype=float
    )

    return float(
        np.mean(
            np.abs(
                predicted_array
                - true_array
            )
        )
    )


def evaluate_routes_on_matrix(
    routes,
    cost_matrix
) -> float:

    return float(
        calculate_routes_cost(
            routes,
            cost_matrix
        )
    )


def calculate_improvement_percent(
    baseline_cost: float,
    candidate_cost: float
) -> float:

    if baseline_cost <= 0:
        raise ValueError(
            "Baseline cost must be positive."
        )

    return (
        (
            baseline_cost
            - candidate_cost
        )
        /
        baseline_cost
        * 100
    )


def calculate_regret_percent(
    candidate_cost: float,
    oracle_cost: float
) -> float:

    if oracle_cost <= 0:
        raise ValueError(
            "Oracle cost must be positive."
        )

    regret = (
        (
            candidate_cost
            - oracle_cost
        )
        /
        oracle_cost
        * 100
    )

    if abs(regret) < 1e-9:
        regret = 0.0

    return regret

