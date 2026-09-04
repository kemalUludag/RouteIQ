import pandas as pd

from src.data_generator import (
    generate_cvrp_instance_model
)
from src.bruteforce_solver import (
    BruteForceCVRPSolver
)
from src.ortools_solver import (
    ORToolsCVRPSolver
)
from src.experiment_runner import (
    ExperimentRunner
)


customer_sizes = [
    5,
    6,
    7,
    8,
    9,
    10,
    20,
    50
]

seeds = range(5)

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
    BruteForceCVRPSolver(
        max_customers=9
    ),

    ORToolsCVRPSolver(
        time_limit_seconds=1
    )
]


runner = ExperimentRunner(
    solvers=solvers
)

results_df = runner.run(
    instances=instances
)


results_df.to_csv(
    "data/experiment_results.csv",
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
        max_runtime=(
            "runtime_seconds",
            "max"
        ),
        feasibility_rate=(
            "feasible",
            "mean"
        ),
        mean_gap=(
            "optimality_gap_percent",
            "mean"
        ),
        max_gap=(
            "optimality_gap_percent",
            "max"
        )
    )
    .reset_index()
)


summary_df.to_csv(
    "data/experiment_summary.csv",
    index=False
)


print("=== EXPERIMENT SUMMARY ===")
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
    "data/experiment_results.csv"
)

print(
    "data/experiment_summary.csv"
)

