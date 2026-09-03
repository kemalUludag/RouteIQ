import pandas as pd

from src.problem_data import create_default_instance
from src.bruteforce_solver import BruteForceCVRPSolver
from src.ortools_solver import ORToolsCVRPSolver
from src.evaluation import compare_solvers


instance = create_default_instance()

solvers = [
    BruteForceCVRPSolver(),
    ORToolsCVRPSolver(
        time_limit_seconds=1
    )
]

comparison_rows, solutions = (
    compare_solvers(
        instance,
        solvers
    )
)

comparison_df = pd.DataFrame(
    comparison_rows
)

print("=== SOLVER COMPARISON ===")
print(
    comparison_df.to_string(
        index=False
    )
)

print()

for solution in solutions:
    print(
        f"=== {solution.solver_name} ==="
    )

    print(
        f"Status: {solution.status}"
    )

    print(
        f"Routes: {solution.routes}"
    )

    print(
        f"Distance: "
        f"{solution.total_distance:.2f}"
    )

    print(
        f"Runtime: "
        f"{solution.runtime_seconds:.6f} s"
    )

    print()

    