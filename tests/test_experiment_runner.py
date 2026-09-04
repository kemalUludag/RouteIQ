from src.models import CVRPInstance
from src.bruteforce_solver import (
    BruteForceCVRPSolver
)
from src.ortools_solver import (
    ORToolsCVRPSolver
)
from src.experiment_runner import (
    ExperimentRunner
)


def test_runner_skips_unsupported_solver():
    points = [
        (float(i), 0.0)
        for i in range(11)
    ]

    demands = [
        0
    ] + [
        1
        for _ in range(10)
    ]

    instance = CVRPInstance(
        name="runner_skip_test",
        points=points,
        demands=demands,
        num_vehicles=2,
        vehicle_capacities=[
            10,
            10
        ]
    )

    solvers = [
        BruteForceCVRPSolver(
            max_customers=9
        ),
        ORToolsCVRPSolver(
            time_limit_seconds=1
        )
    ]

    runner = ExperimentRunner(
        solvers
    )

    results = runner.run(
        [instance]
    )

    assert len(results) == 1

    assert (
        results.iloc[0]["solver"]
        == "OR-Tools"
    )
    