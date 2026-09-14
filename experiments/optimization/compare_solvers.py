import pandas as pd

from routeiq.gurobi_solver import (
    GurobiCVRPSolver
)
from routeiq.problem_data import create_default_instance
from routeiq.bruteforce_solver import BruteForceCVRPSolver
from routeiq.ortools_solver import ORToolsCVRPSolver
from routeiq.evaluation import compare_solvers
from routeiq.clarke_wright_solver import (
    ClarkeWrightCVRPSolver
)

instance = create_default_instance()

solvers = [
    BruteForceCVRPSolver(),

    ClarkeWrightCVRPSolver(),

    GurobiCVRPSolver(
        time_limit_seconds=10,
        output_flag=0,
        max_customers=9
    ),

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

