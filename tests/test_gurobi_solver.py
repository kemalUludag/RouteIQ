import pytest

from src.problem_data import (
    create_default_instance
)
from src.bruteforce_solver import (
    BruteForceCVRPSolver
)
from src.gurobi_solver import (
    GurobiCVRPSolver
)

from src.models import CVRPInstance
from src.routing import calculate_route_demand


def test_gurobi_returns_feasible_solution():
    instance = create_default_instance()

    solver = GurobiCVRPSolver(
        time_limit_seconds=10,
        output_flag=0
    )

    solution = solver.solve(instance)

    assert solution.feasible is True

    assert solution.status in [
        "OPTIMAL",
        "FEASIBLE"
    ]


def test_gurobi_matches_bruteforce_optimum():
    instance = create_default_instance()

    brute_force_solver = (
        BruteForceCVRPSolver()
    )

    gurobi_solver = (
        GurobiCVRPSolver(
            time_limit_seconds=10,
            output_flag=0
        )
    )

    exact_solution = (
        brute_force_solver.solve(
            instance
        )
    )

    gurobi_solution = (
        gurobi_solver.solve(
            instance
        )
    )

    assert (
        gurobi_solution.total_distance
        == pytest.approx(
            exact_solution.total_distance,
            abs=0.01
        )
    )
def test_gurobi_supports_heterogeneous_capacities():
    instance = CVRPInstance(
        name="gurobi_heterogeneous_test",
        points=[
            (0, 0),
            (1, 0),
            (2, 0)
        ],
        demands=[
            0,
            8,
            4
        ],
        num_vehicles=2,
        vehicle_capacities=[
            5,
            10
        ]
    )

    solver = GurobiCVRPSolver(
        time_limit_seconds=10,
        output_flag=0
    )

    solution = solver.solve(instance)

    assert solution.feasible is True
    assert solution.status == "OPTIMAL"

    for route, capacity in zip(
        solution.routes,
        instance.vehicle_capacities
    ):
        route_demand = (
            calculate_route_demand(
                route,
                instance.demands
            )
        )

        assert route_demand <= capacity

    assert 1 in solution.routes[1]


def test_gurobi_allows_unused_vehicle():
    instance = CVRPInstance(
        name="unused_vehicle_test",
        points=[
            (0, 0),
            (2, 0)
        ],
        demands=[
            0,
            3
        ],
        num_vehicles=2,
        vehicle_capacities=[
            10,
            10
        ]
    )

    solver = GurobiCVRPSolver(
        time_limit_seconds=10,
        output_flag=0
    )

    solution = solver.solve(instance)

    assert solution.feasible is True
    assert solution.status == "OPTIMAL"

    unused_routes = [
        route
        for route in solution.routes
        if route == [0, 0]
    ]

    assert len(unused_routes) == 1

        