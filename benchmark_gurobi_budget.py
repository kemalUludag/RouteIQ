import pandas as pd

from src.data_generator import (
    generate_cvrp_instance_model
)
from src.gurobi_solver import (
    GurobiCVRPSolver
)


CUSTOMER_SIZES = [
    10,
    15,
    20
]

SEEDS = range(10)

TIME_LIMITS = [
    0.01,
    0.05,
    0.10,
    0.50,
    1.00
]

VEHICLE_CAPACITY = 15


def main():

    results = []

    for num_customers in CUSTOMER_SIZES:

        num_vehicles = max(
            2,
            (num_customers + 4) // 5
        )

        for seed in SEEDS:

            instance = (
                generate_cvrp_instance_model(
                    num_customers=num_customers,
                    num_vehicles=num_vehicles,
                    vehicle_capacity=VEHICLE_CAPACITY,
                    seed=seed
                )
            )

            for time_limit in TIME_LIMITS:

                print(
                    f"Running "
                    f"n={num_customers}, "
                    f"seed={seed}, "
                    f"time={time_limit}s"
                )

                solver = (
                    GurobiCVRPSolver(
                        time_limit_seconds=time_limit,
                        output_flag=0,
                        max_customers=20
                    )
                )

                solution = (
                    solver.solve(
                        instance
                    )
                )

                metadata = (
                    solution.metadata
                )

                mip_gap = metadata.get(
                    "mip_gap"
                )

                if mip_gap is not None:
                    mip_gap_percent = (
                        mip_gap * 100
                    )
                else:
                    mip_gap_percent = None

                results.append(
                    {
                        "instance":
                            instance.name,

                        "num_customers":
                            num_customers,

                        "seed":
                            seed,

                        "num_vehicles":
                            num_vehicles,

                        "time_limit_seconds":
                            time_limit,

                        "status":
                            solution.status,

                        "feasible":
                            solution.feasible,

                        "distance":
                            (
                                solution.total_distance
                                if solution.feasible
                                else None
                            ),

                        "runtime_seconds":
                            solution.runtime_seconds,

                        "best_bound":
                            metadata.get(
                                "best_bound"
                            ),

                        "mip_gap_percent":
                            mip_gap_percent,

                        "node_count":
                            metadata.get(
                                "node_count"
                            ),

                        "solution_count":
                            metadata.get(
                                "solution_count"
                            ),

                        "gurobi_status":
                            metadata.get(
                                "gurobi_status"
                            )
                    }
                )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        "data/gurobi_budget_results.csv",
        index=False
    )

    # ==============================================
    # Summary
    # ==============================================

    results_df[
        "optimal"
    ] = (
        results_df["status"]
        == "OPTIMAL"
    )

    summary_df = (
        results_df
        .groupby(
            [
                "num_customers",
                "time_limit_seconds"
            ]
        )
        .agg(
            instances=(
                "instance",
                "count"
            ),

            feasible_rate=(
                "feasible",
                "mean"
            ),

            optimal_rate=(
                "optimal",
                "mean"
            ),

            mean_distance=(
                "distance",
                "mean"
            ),

            mean_runtime=(
                "runtime_seconds",
                "mean"
            ),

            mean_mip_gap_percent=(
                "mip_gap_percent",
                "mean"
            ),

            median_mip_gap_percent=(
                "mip_gap_percent",
                "median"
            ),

            max_mip_gap_percent=(
                "mip_gap_percent",
                "max"
            ),

            mean_node_count=(
                "node_count",
                "mean"
            )
        )
        .reset_index()
    )

    summary_df.to_csv(
        "data/gurobi_budget_summary.csv",
        index=False
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "GUROBI BUDGET EXPERIMENT SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        summary_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()

    