import pytest

from src.models import CVRPInstance
from src.problem_data import create_default_instance
from src.ortools_solver import ORToolsCVRPSolver
from src.validation import validate_solution
from src.models import CVRPInstance
from src.routing import calculate_route_demand


def test_ortools_solver_returns_solution():
    instance = create_default_instance()

    solver = ORToolsCVRPSolver(
        time_limit_seconds=1
    )

    solution = solver.solve(instance)

    assert solution.solver_name == "OR-Tools"
    assert solution.feasible is True
    assert solution.status == "FEASIBLE"

    assert len(solution.routes) == (
        instance.num_vehicles
    )

    assert solution.total_distance > 0
    assert solution.runtime_seconds > 0


def test_ortools_solution_is_valid():
    instance = create_default_instance()

    solver = ORToolsCVRPSolver(
        time_limit_seconds=1
    )

    solution = solver.solve(instance)

    is_valid, _ = validate_solution(
        solution.routes,
        instance.num_customers,
        instance.demands,
        max(instance.vehicle_capacities),
        instance.num_vehicles
    )

    assert is_valid is True

def test_ortools_solver_supports_heterogeneous_capacities():
    instance = CVRPInstance(
        name="heterogeneous_fleet_test",
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

    solver = ORToolsCVRPSolver(
        time_limit_seconds=1
    )

    solution = solver.solve(instance)

    assert solution.feasible is True

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

def test_solver_wrapper_rejects_invalid_internal_solution(
    monkeypatch
):
    instance = create_default_instance()

    invalid_routes = [
        [0, 1, 2, 0],
        [0, 3, 4, 0]
    ]

    def fake_solver(*args, **kwargs):
        return invalid_routes

    monkeypatch.setattr(
        "src.ortools_solver._solve_cvrp_routes",
        fake_solver
    )

    solver = ORToolsCVRPSolver(
        time_limit_seconds=1
    )

    solution = solver.solve(instance)

    assert solution.feasible is False
    assert solution.status == "INVALID_SOLUTION"

    assert (
        "missing"
        in solution.metadata[
            "validation_message"
        ].lower()
    )

def test_ortools_solves_with_custom_cost_matrix():

    instance = CVRPInstance(
        name="custom_cost_test",
        points=[
            (0, 0),
            (1, 0),
            (2, 0)
        ],
        demands=[
            0,
            1,
            1
        ],
        num_vehicles=1,
        vehicle_capacities=[
            10
        ]
    )

    cost_matrix = [
        [0.0, 2.0, 10.0],
        [2.0, 0.0, 1.0],
        [10.0, 1.0, 0.0]
    ]

    solver = ORToolsCVRPSolver(
        time_limit_seconds=1
    )

    solution = (
        solver.solve_with_cost_matrix(
            instance=instance,
            cost_matrix=cost_matrix,
            objective_name=(
                "travel_time"
            ),
            objective_unit=(
                "minutes"
            )
        )
    )

    assert solution.feasible is True

    assert solution.metadata[
        "objective_name"
    ] == "travel_time"

    assert solution.metadata[
        "objective_unit"
    ] == "minutes"

    assert solution.total_distance == pytest.approx(
        13.0
    )

def test_ortools_rejects_invalid_cost_matrix():

    instance = CVRPInstance(
        name="invalid_matrix_test",
        points=[
            (0, 0),
            (1, 0),
            (2, 0)
        ],
        demands=[
            0,
            1,
            1
        ],
        num_vehicles=1,
        vehicle_capacities=[
            10
        ]
    )

    invalid_matrix = [
        [0.0, 1.0],
        [1.0, 0.0]
    ]

    solver = ORToolsCVRPSolver(
        time_limit_seconds=1
    )

    with pytest.raises(
        ValueError
    ):
        solver.solve_with_cost_matrix(
            instance,
            invalid_matrix
        )


