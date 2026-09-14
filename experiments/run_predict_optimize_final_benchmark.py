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
    build_free_flow_travel_time_matrix,
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
    30
)


SCENARIOS = [
    {
        "name":
            "weekday_midday",

        "hour":
            12,

        "weekday":
            2
    },

    {
        "name":
            "weekday_rush",

        "hour":
            18,

        "weekday":
            2
    },

    {
        "name":
            "weekend_midday",

        "hour":
            12,

        "weekday":
            5
    },

    {
        "name":
            "weekend_evening",

        "hour":
            19,

        "weekday":
            5
    }
]


WEATHER = "clear"


def main():

    # ==================================================
    # 1. Train finalized prediction model ONCE
    # ==================================================

    print(
        "Training final travel-time model..."
    )

    ml_model = (
        train_final_travel_time_model()
    )

    # ==================================================
    # 2. Optimization engines
    # ==================================================

    # Distance / free-flow / ML all use the
    # same OR-Tools configuration.
    #
    # This controls for optimization-algorithm
    # differences between these three strategies.

    ortools_solver = (
        ORToolsCVRPSolver(
            time_limit_seconds=1
        )
    )

    # Exact true-cost oracle.

    oracle_solver = (
        GurobiCVRPSolver(
            time_limit_seconds=10,
            output_flag=0,
            max_customers=10
        )
    )

    # ==================================================
    # 3. Storage
    # ==================================================

    results = []

    # ==================================================
    # 4. Network loop
    # ==================================================

    for network_seed in NETWORK_SEEDS:

        print(
            "\n"
            "========================================"
        )

        print(
            f"NETWORK {network_seed + 1}/"
            f"{len(NETWORK_SEEDS)} "
            f"(seed={network_seed})"
        )

        print(
            "========================================"
        )

        # ----------------------------------------------
        # Generate one physical CVRP network
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

                seed=(
                    network_seed
                )
            )
        )

        # ----------------------------------------------
        # Persistent road structure
        #
        # Same road network for every scenario
        # belonging to this CVRP instance.
        # ----------------------------------------------

        road_seed = (
            5000
            + network_seed
        )

        # ==================================================
        # 5. Distance-only baseline
        #
        # Distance is independent of traffic scenario,
        # so solve only once per network.
        # ==================================================

        distance_matrix = (
            create_distance_matrix(
                instance.points
            )
        )

        distance_solution = (
            ortools_solver.solve(
                instance
            )
        )

        if not distance_solution.feasible:

            raise RuntimeError(
                "Distance baseline failed for "
                f"{instance.name}."
            )

        # ==================================================
        # 6. Free-flow baseline
        #
        # Road structure is also persistent.
        #
        # We generate one context only to obtain the
        # network's road_type_matrix. Traffic values
        # are ignored by the free-flow calculation.
        # ==================================================

        structural_scenario = (
            generate_routing_context(
                instance=instance,
                hour=12,
                weekday=2,
                weather=WEATHER,
                road_seed=road_seed,
                traffic_seed=90000 + network_seed
            )
        )

        free_flow_matrix = (
            build_free_flow_travel_time_matrix(
                instance=instance,
                scenario=structural_scenario
            )
        )

        free_flow_solution = (
            ortools_solver.solve_with_cost_matrix(
                instance=instance,

                cost_matrix=(
                    free_flow_matrix
                ),

                objective_name=(
                    "free_flow_travel_time"
                ),

                objective_unit=(
                    "minutes"
                )
            )
        )

        if not free_flow_solution.feasible:

            raise RuntimeError(
                "Free-flow baseline failed for "
                f"{instance.name}."
            )

        # ==================================================
        # 7. Dynamic scenario loop
        # ==================================================

        for scenario_index, scenario_config in enumerate(
            SCENARIOS
        ):

            scenario_name = (
                scenario_config[
                    "name"
                ]
            )

            print(
                f"\nScenario: "
                f"{scenario_name}"
            )

            # ------------------------------------------
            # One reproducible dynamic traffic
            # realization for this network + regime.
            #
            # Different scenario regimes receive
            # different deterministic traffic seeds.
            # ------------------------------------------

            traffic_seed = (
                10000
                + network_seed * 100
                + scenario_index
            )

            scenario = (
                generate_routing_context(
                    instance=instance,

                    hour=(
                        scenario_config[
                            "hour"
                        ]
                    ),

                    weekday=(
                        scenario_config[
                            "weekday"
                        ]
                    ),

                    weather=(
                        WEATHER
                    ),

                    road_seed=(
                        road_seed
                    ),

                    traffic_seed=(
                        traffic_seed
                    )
                )
            )

            # ==================================================
            # 8. Dynamic cost matrices
            # ==================================================

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

            # ==================================================
            # 9. Edge-level prediction quality
            # ==================================================

            edge_mae = (
                calculate_matrix_mae(
                    predicted_matrix,
                    true_matrix
                )
            )

            # ==================================================
            # 10. ML-assisted routing decision
            # ==================================================

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
                    "ML routing failed for "
                    f"{instance.name} / "
                    f"{scenario_name}."
                )

            # ==================================================
            # 11. Exact true-cost oracle
            # ==================================================

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
            # Never silently treat an unproven
            # incumbent as the oracle.
            # ------------------------------------------

            if (
                oracle_solution.status
                != "OPTIMAL"
            ):

                raise RuntimeError(
                    "Exact oracle was not proven "
                    "optimal for "
                    f"{instance.name} / "
                    f"{scenario_name}. "
                    f"Status="
                    f"{oracle_solution.status}"
                )

            # ==================================================
            # 12. FAIR evaluation
            #
            # All four routes are evaluated using
            # exactly the same true dynamic matrix.
            # ==================================================

            distance_true_cost = (
                evaluate_routes_on_matrix(
                    distance_solution.routes,
                    true_matrix
                )
            )

            free_flow_true_cost = (
                evaluate_routes_on_matrix(
                    free_flow_solution.routes,
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

            # ==================================================
            # 13. Improvement metrics
            # ==================================================

            ml_improvement_vs_distance = (
                calculate_improvement_percent(
                    baseline_cost=(
                        distance_true_cost
                    ),

                    candidate_cost=(
                        ml_true_cost
                    )
                )
            )

            free_flow_improvement_vs_distance = (
                calculate_improvement_percent(
                    baseline_cost=(
                        distance_true_cost
                    ),

                    candidate_cost=(
                        free_flow_true_cost
                    )
                )
            )

            ml_improvement_vs_free_flow = (
                calculate_improvement_percent(
                    baseline_cost=(
                        free_flow_true_cost
                    ),

                    candidate_cost=(
                        ml_true_cost
                    )
                )
            )

            # ==================================================
            # 14. Oracle regret
            # ==================================================

            distance_regret = (
                calculate_regret_percent(
                    candidate_cost=(
                        distance_true_cost
                    ),

                    oracle_cost=(
                        oracle_true_cost
                    )
                )
            )

            free_flow_regret = (
                calculate_regret_percent(
                    candidate_cost=(
                        free_flow_true_cost
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

            # ==================================================
            # 15. Win / tie / loss against both baselines
            # ==================================================

            tolerance = 1e-9

            if (
                ml_improvement_vs_distance
                > tolerance
            ):

                outcome_vs_distance = (
                    "win"
                )

            elif (
                ml_improvement_vs_distance
                < -tolerance
            ):

                outcome_vs_distance = (
                    "loss"
                )

            else:

                outcome_vs_distance = (
                    "tie"
                )

            if (
                ml_improvement_vs_free_flow
                > tolerance
            ):

                outcome_vs_free_flow = (
                    "win"
                )

            elif (
                ml_improvement_vs_free_flow
                < -tolerance
            ):

                outcome_vs_free_flow = (
                    "loss"
                )

            else:

                outcome_vs_free_flow = (
                    "tie"
                )

            # ==================================================
            # 16. Store one paired scenario
            # ==================================================

            results.append(
                {
                    "instance":
                        instance.name,

                    "network_seed":
                        network_seed,

                    "scenario":
                        scenario_name,

                    "scenario_index":
                        scenario_index,

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

                    # ----------------------------------
                    # Prediction quality
                    # ----------------------------------

                    "edge_prediction_mae":
                        edge_mae,

                    # ----------------------------------
                    # True route costs
                    # ----------------------------------

                    "distance_true_cost":
                        distance_true_cost,

                    "free_flow_true_cost":
                        free_flow_true_cost,

                    "ml_true_cost":
                        ml_true_cost,

                    "oracle_true_cost":
                        oracle_true_cost,

                    # ----------------------------------
                    # Improvements
                    # ----------------------------------

                    "free_flow_improvement_vs_distance_percent":
                        free_flow_improvement_vs_distance,

                    "ml_improvement_vs_distance_percent":
                        ml_improvement_vs_distance,

                    "ml_improvement_vs_free_flow_percent":
                        ml_improvement_vs_free_flow,

                    # ----------------------------------
                    # Exact-oracle regrets
                    # ----------------------------------

                    "distance_regret_vs_oracle_percent":
                        distance_regret,

                    "free_flow_regret_vs_oracle_percent":
                        free_flow_regret,

                    "ml_regret_vs_oracle_percent":
                        ml_regret,

                    # ----------------------------------
                    # Pairwise outcomes
                    # ----------------------------------

                    "ml_outcome_vs_distance":
                        outcome_vs_distance,

                    "ml_outcome_vs_free_flow":
                        outcome_vs_free_flow,

                    # ----------------------------------
                    # Routes
                    # ----------------------------------

                    "distance_routes":
                        json.dumps(
                            distance_solution.routes
                        ),

                    "free_flow_routes":
                        json.dumps(
                            free_flow_solution.routes
                        ),

                    "ml_routes":
                        json.dumps(
                            ml_solution.routes
                        ),

                    "oracle_routes":
                        json.dumps(
                            oracle_solution.routes
                        ),

                    # ----------------------------------
                    # Solver runtimes
                    # ----------------------------------

                    "distance_runtime_seconds":
                        distance_solution.runtime_seconds,

                    "free_flow_runtime_seconds":
                        free_flow_solution.runtime_seconds,

                    "ml_runtime_seconds":
                        ml_solution.runtime_seconds,

                    "oracle_runtime_seconds":
                        oracle_solution.runtime_seconds,

                    # ----------------------------------
                    # Oracle certificate
                    # ----------------------------------

                    "oracle_mip_gap":
                        oracle_solution.metadata.get(
                            "mip_gap"
                        )
                }
            )

            # ==================================================
            # 17. Scenario report
            # ==================================================

            print(
                f"Edge MAE: "
                f"{edge_mae:.3f} min"
            )

            print(
                f"Distance:  "
                f"{distance_true_cost:.3f} min"
            )

            print(
                f"Free flow: "
                f"{free_flow_true_cost:.3f} min"
            )

            print(
                f"ML:        "
                f"{ml_true_cost:.3f} min"
            )

            print(
                f"Oracle:    "
                f"{oracle_true_cost:.3f} min"
            )

            print(
                f"ML vs distance:  "
                f"{ml_improvement_vs_distance:.2f}%"
            )

            print(
                f"ML vs free flow: "
                f"{ml_improvement_vs_free_flow:.2f}%"
            )

            print(
                f"ML oracle regret:"
                f" {ml_regret:.2f}%"
            )

            print(
                "Outcomes: "
                f"distance={outcome_vs_distance}, "
                f"free_flow={outcome_vs_free_flow}"
            )

    # ==================================================
    # 18. Save raw benchmark
    # ==================================================

    results_df = (
        pd.DataFrame(
            results
        )
    )

    results_df.to_csv(
        "data/predict_optimize_final_results.csv",
        index=False
    )

    # ==================================================
    # 19. Basic descriptive summary
    # ==================================================

    distance_wins = (
        results_df[
            "ml_outcome_vs_distance"
        ]
        .eq("win")
        .sum()
    )

    distance_ties = (
        results_df[
            "ml_outcome_vs_distance"
        ]
        .eq("tie")
        .sum()
    )

    distance_losses = (
        results_df[
            "ml_outcome_vs_distance"
        ]
        .eq("loss")
        .sum()
    )

    free_flow_wins = (
        results_df[
            "ml_outcome_vs_free_flow"
        ]
        .eq("win")
        .sum()
    )

    free_flow_ties = (
        results_df[
            "ml_outcome_vs_free_flow"
        ]
        .eq("tie")
        .sum()
    )

    free_flow_losses = (
        results_df[
            "ml_outcome_vs_free_flow"
        ]
        .eq("loss")
        .sum()
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "FINAL PREDICT → OPTIMIZE "
        "BENCHMARK COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"Independent networks: "
        f"{results_df['network_seed'].nunique()}"
    )

    print(
        f"Decision scenarios: "
        f"{len(results_df)}"
    )

    print(
        "\nML vs Distance:"
    )

    print(
        f"Wins:   {distance_wins}"
    )

    print(
        f"Ties:   {distance_ties}"
    )

    print(
        f"Losses: {distance_losses}"
    )

    print(
        "\nML vs Free Flow:"
    )

    print(
        f"Wins:   {free_flow_wins}"
    )

    print(
        f"Ties:   {free_flow_ties}"
    )

    print(
        f"Losses: {free_flow_losses}"
    )

    print(
        "\nMean edge prediction MAE:"
    )

    print(
        f"{results_df['edge_prediction_mae'].mean():.3f} min"
    )

    print(
        "\nMean ML improvement vs distance:"
    )

    print(
        f"{results_df['ml_improvement_vs_distance_percent'].mean():.3f}%"
    )

    print(
        "\nMean ML improvement vs free flow:"
    )

    print(
        f"{results_df['ml_improvement_vs_free_flow_percent'].mean():.3f}%"
    )

    print(
        "\nMean oracle regrets:"
    )

    print(
        "Distance: "
        f"{results_df['distance_regret_vs_oracle_percent'].mean():.3f}%"
    )

    print(
        "Free flow: "
        f"{results_df['free_flow_regret_vs_oracle_percent'].mean():.3f}%"
    )

    print(
        "ML: "
        f"{results_df['ml_regret_vs_oracle_percent'].mean():.3f}%"
    )

    print(
        "\nMaximum ML oracle regret:"
    )

    print(
        f"{results_df['ml_regret_vs_oracle_percent'].max():.3f}%"
    )

    print(
        "\nSaved:"
    )

    print(
        "data/predict_optimize_final_results.csv"
    )


if __name__ == "__main__":
    main()

    