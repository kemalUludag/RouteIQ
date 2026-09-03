import pytest

from src.models import CVRPInstance
from src.problem_data import create_default_instance
from src.bruteforce_solver import BruteForceCVRPSolver
from src.ortools_solver import ORToolsCVRPSolver


def test_bruteforce_solver_returns_optimal_solution():
    instance = create_default_instance()

    solver = BruteForceCVRPSolver()

    solution = solver.solve(instance)

    assert solution.feasible is True
    assert solution.status == "OPTIMAL"
    assert solution.solver_name == "Brute Force"

    assert solution.total_distance > 0

    assert (
        solution.metadata["exact"]
        is True
    )


def test_bruteforce_supports_heterogeneous_capacities():
    instance = CVRPInstance(
        name="heterogeneous_exact_test",
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

    solver = BruteForceCVRPSolver()

    solution = solver.solve(instance)

    assert solution.feasible is True
    assert solution.status == "OPTIMAL"

    assert 2 in solution.routes[0]
    assert 1 in solution.routes[1]


def test_ortools_matches_exact_solution_on_small_instance():
    instance = create_default_instance()

    exact_solver = BruteForceCVRPSolver()

    ortools_solver = ORToolsCVRPSolver(
        time_limit_seconds=1
    )

    exact_solution = (
        exact_solver.solve(instance)
    )

    ortools_solution = (
        ortools_solver.solve(instance)
    )

    assert (
        ortools_solution.total_distance
        == pytest.approx(
            exact_solution.total_distance,
            abs=0.01
        )
    )

    