import pandas as pd

from routeiq.data_generator import (
    generate_cvrp_instance_model
)
from routeiq.clarke_wright_solver import (
    ClarkeWrightCVRPSolver
)
from routeiq.ortools_solver import (
    ORToolsCVRPSolver
)
from routeiq.experiment_runner import (
    ExperimentRunner
)


customer_sizes = [
    20,
    50
]

seeds = range(30)

instances = []

for num_customers in customer_sizes:
    num_vehicles = max(
        2,
        (num_customers + 4) // 5
    )

    for seed in seeds:
        instance = (
            generate_cvrp_instance_model(
                num_customers=num_customers,
                num_vehicles=num_vehicles,
                vehicle_capacity=15,
                seed=seed
            )
        )

        instances.append(instance)


solvers = [
    ClarkeWrightCVRPSolver(),

    ORToolsCVRPSolver(
        time_limit_seconds=1,
        solver_name="OR-Tools-1s"
    ),

    ORToolsCVRPSolver(
        time_limit_seconds=3,
        solver_name="OR-Tools-3s"
    ),

    ORToolsCVRPSolver(
        time_limit_seconds=5,
        solver_name="OR-Tools-5s"
    )
]


runner = ExperimentRunner(
    solvers=solvers
)

results_df = runner.run(
    instances=instances
)


results_df.to_csv(
    "results/optimization/budget_experiment_results.csv",
    index=False
)


summary_df = (
    results_df
    .groupby([
        "num_customers",
        "solver"
    ])
    .agg(
        instances=(
            "instance",
            "count"
        ),
        mean_distance=(
            "distance",
            "mean"
        ),
        std_distance=(
            "distance",
            "std"
        ),
        mean_runtime=(
            "runtime_seconds",
            "mean"
        ),
        feasibility_rate=(
            "feasible",
            "mean"
        )
    )
    .reset_index()
)


summary_df.to_csv(
    "results/optimization/budget_experiment_summary.csv",
    index=False
)


print(
    "=== BUDGET EXPERIMENT SUMMARY ==="
)

print(
    summary_df.to_string(
        index=False
    )
)

print()

print(
    "Saved:"
)

print(
    "results/optimization/budget_experiment_results.csv"
)

print(
    "results/optimization/budget_experiment_summary.csv"
)
