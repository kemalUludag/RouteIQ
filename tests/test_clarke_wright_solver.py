from src.models import CVRPInstance
from src.problem_data import create_default_instance
from src.clarke_wright_solver import (
    ClarkeWrightCVRPSolver
)
from src.validation import validate_cvrp_routes


def test_clarke_wright_returns_feasible_solution():
    instance = create_default_instance()

    solver = ClarkeWrightCVRPSolver()

    solution = solver.solve(instance)

    assert solution.feasible is True
    assert solution.status == "FEASIBLE"

    is_valid, _ = validate_cvrp_routes(
        instance,
        solution.routes
    )

    assert is_valid is True


def test_clarke_wright_visits_all_customers():
    instance = create_default_instance()

    solver = ClarkeWrightCVRPSolver()

    solution = solver.solve(instance)

    visited_customers = []

    for route in solution.routes:
        visited_customers.extend(
            route[1:-1]
        )

    assert sorted(
        visited_customers
    ) == list(
        range(
            1,
            instance.num_customers + 1
        )
    )


def test_clarke_wright_rejects_heterogeneous_fleet():
    instance = CVRPInstance(
        name="heterogeneous_test",
        points=[
            (0, 0),
            (1, 0),
            (2, 0)
        ],
        demands=[
            0,
            2,
            3
        ],
        num_vehicles=2,
        vehicle_capacities=[
            5,
            10
        ]
    )

    solver = ClarkeWrightCVRPSolver()

    assert solver.supports(
        instance
    ) is False

    solution = solver.solve(
        instance
    )

    assert (
        solution.status
        == "UNSUPPORTED_INSTANCE"
    )
    