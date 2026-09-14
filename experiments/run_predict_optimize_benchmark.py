import json

import pandas as pd

from src.data_generator import (
    generate_cvrp_instance_model
)

from src.distance import (
    create_distance_matrix
)

from src.gurobi_solver import (
    GurobiCVRPSolver
)

from src.ml_training import (
    train_final_travel_time_model
)

from src.ortools_solver import (
    ORToolsCVRPSolver
)

from src.predict_optimize_evaluation import (
    calculate_matrix_mae,
    evaluate_routes_on_matrix,
    calculate_improvement_percent,
    calculate_regret_percent
)

from src.travel_time_matrix import (
    generate_routing_context,
    build_predicted_travel_time_matrix,
    build_true_expected_travel_time_matrix
)


# ==================================================
# Experiment configuration
# ==================================================

NUM_CUSTOMERS = 10

NUM_VEHICLES = 2

VEHICLE_CAPACITY = 15

NETWORK_SEEDS = range(
    5
)


ENVIRONMENTS = [
    {
        "name":
            "weekday_midday_clear",

        "hour":
            12,

        "weekday":
            2,

        "weather":
            "clear",

        "traffic_profile":
            1
    },

    {
        "name":
            "weekday_rush_clear",

        "hour":
            18,

        "weekday":
            2,

        "weather":
            "clear",

        "traffic_profile":
            2
    },

    {
        "name":
            "weekday_rush_rain",

        "hour":
            18,

        "weekday":
            2,

        "weather":
            "rain",

        "traffic_profile":
            2
    },

    {
        "name":
            "weekday_rush_heavy_rain",

        "hour":
            18,

        "weekday":
            2,

        "weather":
            "heavy_rain",

        "traffic_profile":
            2
    },

    {
        "name":
            "weekend_midday_clear",

        "hour":
            12,

        "weekday":
            5,

        "weather":
            "clear",

        "traffic_profile":
            3
    },

    {
        "name":
            "weekend_evening_rain",

        "hour":
            19,

        "weekday":
            5,

        "weather":
            "rain",

        "traffic_profile":
            4
    }
]


def main():

    # ==================================================
    # Train finalized ML predictor ONCE
    # ==================================================

    print(
        "Training final travel-time model..."
    )

    ml_model = (
        train_final_travel_time_model()
    )

    # ==================================================
    # Solvers
    # ==================================================

    # Same heuristic optimizer is used for
    # Static and ML decisions.

    ortools_solver = (
        ORToolsCVRPSolver(
            time_limit_seconds=1
        )
    )

    # Gurobi is used only for the true-cost oracle.
    #
    # n=10 is comfortably within our current
    # restricted-license experiment size.

    oracle_solver = (
        GurobiCVRPSolver(
            time_limit_seconds=10,
            output_flag=0,
            max_customers=10
        )
    )

    results = []

    # ==================================================
    # Experiment loop
    # ==================================================

    for network_seed in NETWORK_SEEDS:

        print(
            "\n"
            "========================================"
        )

        print(
            f"NETWORK SEED {network_seed}"
        )

        print(
            "========================================"
        )

        # ----------------------------------------------
        # One physical CVRP network
        # ----------------------------------------------

        instance = (
            generate_cvrp_instance_model(
                num_customers=(
                    NUM_CUSTOMERS
                ),
                num_vehicles=(
                    NUM_VEHICLES
                ),
                vehicle_capacity=(
                    VEHICLE_CAPACITY
                ),
                seed=network_seed
            )
        )

        # ----------------------------------------------
        # Persistent road structure
        #
        # Same road seed for every environment of
        # this network.
        # ----------------------------------------------

        road_seed = (
            5000
            + network_seed
        )

        # ----------------------------------------------
        # Static distance decision needs to be solved
        # only once per physical network.
        #
        # Weather / traffic do not affect it.
        # ----------------------------------------------

        distance_matrix = (
            create_distance_matrix(
                instance.points
            )
        )

        static_solution = (
            ortools_solver.solve(
                instance
            )
        )

        if not static_solution.feasible:
            raise RuntimeError(
                "Static OR-Tools solution "
                f"was infeasible for {instance.name}."
            )

        # ==================================================
        # Environment loop
        # ==================================================

        for environment in ENVIRONMENTS:

            environment_name = (
                environment["name"]
            )

            print(
                f"\nRunning: "
                f"{environment_name}"
            )

            # ------------------------------------------
            # Deterministic traffic seed
            # ------------------------------------------
            #
            # Notice:
            #
            # rush_clear / rush_rain /
            # rush_heavy_rain share the same
            # traffic_profile.
            #
            # Therefore they use the SAME traffic
            # matrix and differ only in weather.
            # ------------------------------------------

            traffic_seed = (
                10000
                + network_seed * 100
                + environment[
                    "traffic_profile"
                ]
            )

            scenario = (
                generate_routing_context(
                    instance=instance,

                    hour=(
                        environment[
                            "hour"
                        ]
                    ),

                    weekday=(
                        environment[
                            "weekday"
                        ]
                    ),

                    weather=(
                        environment[
                            "weather"
                        ]
                    ),

                    road_seed=(
                        road_seed
                    ),

                    traffic_seed=(
                        traffic_seed
                    )
                )
            )

            # ==========================================
            # Cost matrices
            # ==========================================

            predicted_matrix = (
                build_predicted_travel_time_matrix(
                    instance=instance,
                    scenario=scenario,
                    model=ml_model
                )
            )

            true_matrix = (
                build_true_expected_travel_time_matrix(
                    instance=instance,
                    scenario=scenario
                )
            )

            # ==========================================
            # Prediction quality
            # ==========================================

            edge_mae = (
                calculate_matrix_mae(
                    predicted_matrix,
                    true_matrix
                )
            )

            # ==========================================
            # ML decision
            # ==========================================

            ml_solution = (
                ortools_solver.solve_with_cost_matrix(
                    instance=instance,

                    cost_matrix=(
                        predicted_matrix
                    ),

                    objective_name=(
                        "predicted_travel_time"
                    ),

                    objective_unit=(
                        "minutes"
                    )
                )
            )

            if not ml_solution.feasible:
                raise RuntimeError(
                    "ML OR-Tools solution "
                    f"was infeasible for "
                    f"{instance.name} / "
                    f"{environment_name}."
                )

            # ==========================================
            # Exact Oracle decision
            # ==========================================

            oracle_solution = (
                oracle_solver.solve_with_cost_matrix(
                    instance=instance,

                    cost_matrix=(
                        true_matrix
                    ),

                    objective_name=(
                        "true_expected_travel_time"
                    ),

                    objective_unit=(
                        "minutes"
                    )
                )
            )

            # ------------------------------------------
            # Oracle must REALLY be optimal.
            # ------------------------------------------

            if (
                oracle_solution.status
                != "OPTIMAL"
            ):
                raise RuntimeError(
                    "Oracle solution was not "
                    "proven optimal for "
                    f"{instance.name} / "
                    f"{environment_name}. "
                    f"Status: "
                    f"{oracle_solution.status}"
                )

            # ==========================================
            # FAIR true-cost evaluation
            # ==========================================

            static_true_cost = (
                evaluate_routes_on_matrix(
                    static_solution.routes,
                    true_matrix
                )
            )

            ml_true_cost = (
                evaluate_routes_on_matrix(
                    ml_solution.routes,
                    true_matrix
                )
            )

            oracle_true_cost = (
                evaluate_routes_on_matrix(
                    oracle_solution.routes,
                    true_matrix
                )
            )

            # ==========================================
            # Decision-quality metrics
            # ==========================================

            ml_improvement = (
                calculate_improvement_percent(
                    baseline_cost=(
                        static_true_cost
                    ),

                    candidate_cost=(
                        ml_true_cost
                    )
                )
            )

            static_regret = (
                calculate_regret_percent(
                    candidate_cost=(
                        static_true_cost
                    ),

                    oracle_cost=(
                        oracle_true_cost
                    )
                )
            )

            ml_regret = (
                calculate_regret_percent(
                    candidate_cost=(
                        ml_true_cost
                    ),

                    oracle_cost=(
                        oracle_true_cost
                    )
                )
            )

            # ==========================================
            # Win / tie / loss
            # ==========================================

            tolerance = 1e-9

            if ml_improvement > tolerance:
                outcome = "win"

            elif ml_improvement < -tolerance:
                outcome = "loss"

            else:
                outcome = "tie"

            # ==========================================
            # Record scenario
            # ==========================================

            results.append(
                {
                    "instance":
                        instance.name,

                    "network_seed":
                        network_seed,

                    "environment":
                        environment_name,

                    "hour":
                        scenario.hour,

                    "weekday":
                        scenario.weekday,

                    "weather":
                        scenario.weather,

                    "road_seed":
                        road_seed,

                    "traffic_seed":
                        traffic_seed,

                    "edge_prediction_mae":
                        edge_mae,

                    "static_true_cost":
                        static_true_cost,

                    "ml_true_cost":
                        ml_true_cost,

                    "oracle_true_cost":
                        oracle_true_cost,

                    "ml_improvement_vs_static_percent":
                        ml_improvement,

                    "static_regret_vs_oracle_percent":
                        static_regret,

                    "ml_regret_vs_oracle_percent":
                        ml_regret,

                    "outcome":
                        outcome,

                    "static_routes":
                        json.dumps(
                            static_solution.routes
                        ),

                    "ml_routes":
                        json.dumps(
                            ml_solution.routes
                        ),

                    "oracle_routes":
                        json.dumps(
                            oracle_solution.routes
                        ),

                    "static_runtime_seconds":
                        static_solution.runtime_seconds,

                    "ml_runtime_seconds":
                        ml_solution.runtime_seconds,

                    "oracle_runtime_seconds":
                        oracle_solution.runtime_seconds,

                    "oracle_mip_gap":
                        oracle_solution.metadata.get(
                            "mip_gap"
                        )
                }
            )

            print(
                f"Edge MAE: "
                f"{edge_mae:.3f} min"
            )

            print(
                f"Static: "
                f"{static_true_cost:.3f} min"
            )

            print(
                f"ML:     "
                f"{ml_true_cost:.3f} min"
            )

            print(
                f"Oracle: "
                f"{oracle_true_cost:.3f} min"
            )

            print(
                f"ML improvement: "
                f"{ml_improvement:.2f}%"
            )

            print(
                f"ML regret: "
                f"{ml_regret:.2f}%"
            )

            print(
                f"Outcome: "
                f"{outcome}"
            )

    # ==================================================
    # Save raw experiment results
    # ==================================================

    results_df = (
        pd.DataFrame(
            results
        )
    )

    results_df.to_csv(
        "data/predict_optimize_results.csv",
        index=False
    )

    # ==================================================
    # Basic overall summary
    # ==================================================

    wins = (
        results_df[
            "outcome"
        ]
        .eq("win")
        .sum()
    )

    ties = (
        results_df[
            "outcome"
        ]
        .eq("tie")
        .sum()
    )

    losses = (
        results_df[
            "outcome"
        ]
        .eq("loss")
        .sum()
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "PREDICT → OPTIMIZE BENCHMARK COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"Scenarios: "
        f"{len(results_df)}"
    )

    print(
        f"Wins:   {wins}"
    )

    print(
        f"Ties:   {ties}"
    )

    print(
        f"Losses: {losses}"
    )

    print(
        "\nMean edge prediction MAE:"
    )

    print(
        f"{results_df['edge_prediction_mae'].mean():.3f} min"
    )

    print(
        "\nMean ML improvement vs static:"
    )

    print(
        f"{results_df['ml_improvement_vs_static_percent'].mean():.3f}%"
    )

    print(
        "\nMean static regret vs oracle:"
    )

    print(
        f"{results_df['static_regret_vs_oracle_percent'].mean():.3f}%"
    )

    print(
        "\nMean ML regret vs oracle:"
    )

    print(
        f"{results_df['ml_regret_vs_oracle_percent'].mean():.3f}%"
    )

    print(
        "\nSaved:"
    )

    print(
        "data/predict_optimize_results.csv"
    )


if __name__ == "__main__":
    main()

    