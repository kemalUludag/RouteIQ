import pytest

from routeiq.predict_optimize_evaluation import (
    calculate_matrix_mae,
    evaluate_routes_on_matrix,
    calculate_improvement_percent,
    calculate_regret_percent
)


def test_calculate_matrix_mae():

    predicted = [
        [0.0, 10.0, 20.0],
        [10.0, 0.0, 30.0],
        [20.0, 30.0, 0.0]
    ]

    true = [
        [0.0, 12.0, 18.0],
        [12.0, 0.0, 33.0],
        [18.0, 33.0, 0.0]
    ]

    # Absolute errors:
    #
    # 2, 2, 2, 3, 2, 3
    #
    # Mean = 14 / 6

    mae = calculate_matrix_mae(
        predicted,
        true
    )

    assert mae == pytest.approx(
        14 / 6
    )


def test_evaluate_routes_on_matrix():

    matrix = [
        [0.0, 5.0, 10.0],
        [5.0, 0.0, 3.0],
        [10.0, 3.0, 0.0]
    ]

    routes = [
        [
            0,
            1,
            2,
            0
        ]
    ]

    cost = evaluate_routes_on_matrix(
        routes,
        matrix
    )

    assert cost == pytest.approx(
        18.0
    )


def test_calculate_improvement_percent():

    improvement = (
        calculate_improvement_percent(
            baseline_cost=200.0,
            candidate_cost=150.0
        )
    )

    assert improvement == pytest.approx(
        25.0
    )


def test_calculate_regret_percent():

    regret = calculate_regret_percent(
        candidate_cost=110.0,
        oracle_cost=100.0
    )

    assert regret == pytest.approx(
        10.0
    )

    