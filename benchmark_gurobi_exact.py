import pandas as pd

from src.data_generator import (
    generate_cvrp_instance_model
)
from src.bruteforce_solver import (
    BruteForceCVRPSolver
)
from src.gurobi_solver import (
    GurobiCVRPSolver
)


CUSTOMER_SIZES = [
    5,
    6,
    7,
    8,
    9
]

SEEDS = range(5)

VEHICLE_CAPACITY = 15


def main():

    brute_force_solver = (
        BruteForceCVRPSolver(
            max_customers=9
        )
    )

    gurobi_solver = (
        GurobiCVRPSolver(
            time_limit_seconds=30,
            output_flag=0,
            max_customers=9
        )
    )

    results = []

    for num_customers in CUSTOMER_SIZES:
        for seed in SEEDS:

            num_vehicles = max(
                2,
                (num_customers + 4) // 5
            )

            instance = (
                generate_cvrp_instance_model(
                    num_customers=num_customers,
                    num_vehicles=num_vehicles,
                    vehicle_capacity=(
                        VEHICLE_CAPACITY
                    ),
                    seed=seed
                )
            )

            print(
                f"\nRunning "
                f"n={num_customers}, "
                f"seed={seed}"
            )

            # ==========================================
            # Brute Force
            # ==========================================

            brute_force_solution = (
                brute_force_solver.solve(
                    instance
                )
            )

            # ==========================================
            # Gurobi MILP
            # ==========================================

            gurobi_solution = (
                gurobi_solver.solve(
                    instance
                )
            )

            # ==========================================
            # Exact objective comparison
            # ==========================================

            if (
                brute_force_solution.feasible
                and gurobi_solution.feasible
            ):
                objective_difference = abs(
                    brute_force_solution.total_distance
                    - gurobi_solution.total_distance
                )

                objective_match = (
                    objective_difference
                    <= 1e-6
                )

            else:
                objective_difference = (
                    float("nan")
                )

                objective_match = False

            # ==========================================
            # Collect Gurobi metadata
            # ==========================================

            gurobi_metadata = (
                gurobi_solution.metadata
            )

            row = {
                "instance":
                    instance.name,

                "num_customers":
                    num_customers,

                "seed":
                    seed,

                "num_vehicles":
                    num_vehicles,

                # Brute Force
                "bruteforce_status":
                    brute_force_solution.status,

                "bruteforce_feasible":
                    brute_force_solution.feasible,

                "bruteforce_distance":
                    brute_force_solution.total_distance,

                "bruteforce_runtime":
                    brute_force_solution.runtime_seconds,

                # Gurobi
                "gurobi_status":
                    gurobi_solution.status,

                "gurobi_feasible":
                    gurobi_solution.feasible,

                "gurobi_distance":
                    gurobi_solution.total_distance,

                "gurobi_runtime":
                    gurobi_solution.runtime_seconds,

                "gurobi_mip_gap":
                    gurobi_metadata.get(
                        "mip_gap"
                    ),

                "gurobi_node_count":
                    gurobi_metadata.get(
                        "node_count"
                    ),

                "gurobi_best_bound":
                    gurobi_metadata.get(
                        "best_bound"
                    ),

                # Exact parity
                "objective_difference":
                    objective_difference,

                "objective_match":
                    objective_match
            }

            results.append(
                row
            )

            print(
                "Brute Force:",
                brute_force_solution.status,
                f"{brute_force_solution.total_distance:.6f}",
                f"{brute_force_solution.runtime_seconds:.6f}s"
            )

            print(
                "Gurobi:",
                gurobi_solution.status,
                f"{gurobi_solution.total_distance:.6f}",
                f"{gurobi_solution.runtime_seconds:.6f}s",
                "match=",
                objective_match
            )

    # ==============================================
    # Full results
    # ==============================================

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        "data/gurobi_exact_results.csv",
        index=False
    )

    # ==============================================
    # Summary by problem size
    # ==============================================

    summary_df = (
        results_df
        .groupby(
            "num_customers"
        )
        .agg(
            instances=(
                "instance",
                "count"
            ),

            objective_matches=(
                "objective_match",
                "sum"
            ),

            match_rate=(
                "objective_match",
                "mean"
            ),

            bruteforce_runtime_mean=(
                "bruteforce_runtime",
                "mean"
            ),

            gurobi_runtime_mean=(
                "gurobi_runtime",
                "mean"
            ),

            gurobi_runtime_max=(
                "gurobi_runtime",
                "max"
            ),

            gurobi_mip_gap_max=(
                "gurobi_mip_gap",
                "max"
            ),

            gurobi_node_count_mean=(
                "gurobi_node_count",
                "mean"
            ),

            gurobi_node_count_max=(
                "gurobi_node_count",
                "max"
            )
        )
        .reset_index()
    )

    summary_df.to_csv(
        "data/gurobi_exact_summary.csv",
        index=False
    )

    # ==============================================
    # Print final report
    # ==============================================

    total_instances = len(
        results_df
    )

    total_matches = int(
        results_df[
            "objective_match"
        ].sum()
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "GUROBI EXACT BENCHMARK SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        f"Exact objective matches: "
        f"{total_matches}/{total_instances}"
    )

    print(
        f"Match rate: "
        f"{100 * total_matches / total_instances:.2f}%"
    )

    print(
        "\nSummary by customer count:"
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    mismatches = (
        results_df[
            ~results_df[
                "objective_match"
            ]
        ]
    )

    if mismatches.empty:
        print(
            "\nAll Gurobi solutions "
            "match the brute-force optimum."
        )

    else:
        print(
            "\nWARNING: Objective mismatches found:"
        )

        print(
            mismatches[
                [
                    "instance",
                    "bruteforce_distance",
                    "gurobi_distance",
                    "objective_difference",
                    "gurobi_status"
                ]
            ].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()


    