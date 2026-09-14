import pytest

from src.routing import (
    calculate_route_cost,
    calculate_routes_cost
)


def test_calculate_route_cost():

    cost_matrix = [
        [0.0, 5.0, 9.0],
        [5.0, 0.0, 4.0],
        [9.0, 4.0, 0.0]
    ]

    route = [
        0,
        1,
        2,
        0
    ]

    # 0 -> 1 = 5
    # 1 -> 2 = 4
    # 2 -> 0 = 9
    #
    # Total = 18

    cost = calculate_route_cost(
        route,
        cost_matrix
    )

    assert cost == pytest.approx(
        18.0
    )


def test_calculate_routes_cost():

    cost_matrix = [
        [0.0, 5.0, 9.0],
        [5.0, 0.0, 4.0],
        [9.0, 4.0, 0.0]
    ]

    routes = [
        [
            0,
            1,
            0
        ],
        [
            0,
            2,
            0
        ]
    ]

    # Route 1:
    # 0 -> 1 -> 0
    # 5 + 5 = 10
    #
    # Route 2:
    # 0 -> 2 -> 0
    # 9 + 9 = 18
    #
    # Fleet total = 28

    cost = calculate_routes_cost(
        routes,
        cost_matrix
    )

    assert cost == pytest.approx(
        28.0
    )


def test_unused_vehicle_has_zero_cost():

    cost_matrix = [
        [0.0, 5.0],
        [5.0, 0.0]
    ]

    route = [
        0,
        0
    ]

    cost = calculate_route_cost(
        route,
        cost_matrix
    )

    assert cost == pytest.approx(
        0.0
    )

    