import json

import numpy as np
import pandas as pd

from routeiq.data_generator import (
    generate_cvrp_instance_model
)

from routeiq.predict_optimize_evaluation import (
    evaluate_routes_on_matrix,
    calculate_improvement_percent
)

from routeiq.travel_time_matrix import (
    generate_routing_context,
    build_realized_travel_time_matrix
)


# ==================================================
# Configuration
# ==================================================

NUM_CUSTOMERS = 10

NUM_VEHICLES = 2

VEHICLE_CAPACITY = 15

NUM_REALIZATIONS = 100

NOISE_STD = 2.5


def main():

    # ==================================================
    # Load the finalized routing decisions
    # ==================================================

    decision_df = pd.read_csv(
        "data/predict_optimize_final_results.csv"
    )

    realization_rows = []

    scenario_summary_rows = []

    # ==================================================
    # Each row represents one fixed routing decision
    # scenario from the final benchmark.
    # ==================================================

    for row_index, row in (
        decision_df.iterrows()
    ):

        network_seed = int(
            row["network_seed"]
        )

        print(
            "\n"
            "========================================"
        )

        print(
            f"Scenario "
            f"{row_index + 1}/"
            f"{len(decision_df)}"
        )

        print(
            f"Network: {network_seed}"
        )

        print(
            f"Regime: {row['scenario']}"
        )

        print(
            "========================================"
        )

        # ==================================================
        # Reconstruct physical CVRP instance
        # ==================================================

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

        # ==================================================
        # Reconstruct exact decision environment
        # ==================================================

        scenario = (
            generate_routing_context(
                instance=instance,

                hour=int(
                    row["hour"]
                ),

                weekday=int(
                    row["weekday"]
                ),

                weather=str(
                    row["weather"]
                ),

                road_seed=int(
                    row["road_seed"]
                ),

                traffic_seed=int(
                    row["traffic_seed"]
                )
            )
        )

        # ==================================================
        # Recover previously selected routes
        #
        # Important:
        # NO reoptimization occurs here.
        # ==================================================

        distance_routes = json.loads(
            row["distance_routes"]
        )

        free_flow_routes = json.loads(
            row["free_flow_routes"]
        )

        ml_routes = json.loads(
            row["ml_routes"]
        )

        oracle_routes = json.loads(
            row["oracle_routes"]
        )

        # ==================================================
        # Store realization-level costs locally
        # ==================================================

        distance_costs = []
        free_flow_costs = []
        ml_costs = []
        oracle_route_costs = []

        ml_vs_free_flow_improvements = []
        ml_vs_distance_improvements = []

        # ==================================================
        # Monte Carlo realizations
        # ==================================================

        for realization_index in range(
            NUM_REALIZATIONS
        ):

            # Unique but reproducible random seed.
            #
            # The SAME matrix is then used for
            # every routing strategy.

            realization_seed = (
                1_000_000
                + row_index * 10_000
                + realization_index
            )

            realized_matrix = (
                build_realized_travel_time_matrix(
                    instance=instance,
                    scenario=scenario,
                    seed=realization_seed,
                    noise_std=NOISE_STD
                )
            )

            # ==========================================
            # Fair realized-cost evaluation
            # ==========================================

            distance_cost = (
                evaluate_routes_on_matrix(
                    distance_routes,
                    realized_matrix
                )
            )

            free_flow_cost = (
                evaluate_routes_on_matrix(
                    free_flow_routes,
                    realized_matrix
                )
            )

            ml_cost = (
                evaluate_routes_on_matrix(
                    ml_routes,
                    realized_matrix
                )
            )

            oracle_route_cost = (
                evaluate_routes_on_matrix(
                    oracle_routes,
                    realized_matrix
                )
            )

            # ==========================================
            # Pairwise improvements
            # ==========================================

            ml_vs_free_flow = (
                calculate_improvement_percent(
                    baseline_cost=(
                        free_flow_cost
                    ),

                    candidate_cost=(
                        ml_cost
                    )
                )
            )

            ml_vs_distance = (
                calculate_improvement_percent(
                    baseline_cost=(
                        distance_cost
                    ),

                    candidate_cost=(
                        ml_cost
                    )
                )
            )

            # ==========================================
            # Local storage
            # ==========================================

            distance_costs.append(
                distance_cost
            )

            free_flow_costs.append(
                free_flow_cost
            )

            ml_costs.append(
                ml_cost
            )

            oracle_route_costs.append(
                oracle_route_cost
            )

            ml_vs_free_flow_improvements.append(
                ml_vs_free_flow
            )

            ml_vs_distance_improvements.append(
                ml_vs_distance
            )

            # ==========================================
            # Raw realization record
            # ==========================================

            realization_rows.append(
                {
                    "network_seed":
                        network_seed,

                    "scenario":
                        row["scenario"],

                    "decision_scenario_index":
                        row_index,

                    "realization_index":
                        realization_index,

                    "realization_seed":
                        realization_seed,

                    "distance_realized_cost":
                        distance_cost,

                    "free_flow_realized_cost":
                        free_flow_cost,

                    "ml_realized_cost":
                        ml_cost,

                    "expected_oracle_route_realized_cost":
                        oracle_route_cost,

                    "ml_improvement_vs_distance_percent":
                        ml_vs_distance,

                    "ml_improvement_vs_free_flow_percent":
                        ml_vs_free_flow,

                    "ml_beats_distance":
                        (
                            ml_cost
                            < distance_cost
                        ),

                    "ml_beats_free_flow":
                        (
                            ml_cost
                            < free_flow_cost
                        )
                }
            )

        # ==================================================
        # Convert scenario arrays
        # ==================================================

        distance_costs = np.asarray(
            distance_costs
        )

        free_flow_costs = np.asarray(
            free_flow_costs
        )

        ml_costs = np.asarray(
            ml_costs
        )

        oracle_route_costs = np.asarray(
            oracle_route_costs
        )

        ml_vs_free_flow_improvements = (
            np.asarray(
                ml_vs_free_flow_improvements
            )
        )

        ml_vs_distance_improvements = (
            np.asarray(
                ml_vs_distance_improvements
            )
        )

        # ==================================================
        # Scenario-level stochastic metrics
        # ==================================================

        mean_distance_cost = float(
            np.mean(
                distance_costs
            )
        )

        mean_free_flow_cost = float(
            np.mean(
                free_flow_costs
            )
        )

        mean_ml_cost = float(
            np.mean(
                ml_costs
            )
        )

        mean_oracle_route_cost = float(
            np.mean(
                oracle_route_costs
            )
        )

        # ------------------------------------------
        # Improvement based on mean realized costs
        # ------------------------------------------

        mean_cost_ml_improvement_vs_free_flow = (
            calculate_improvement_percent(
                baseline_cost=(
                    mean_free_flow_cost
                ),

                candidate_cost=(
                    mean_ml_cost
                )
            )
        )

        mean_cost_ml_improvement_vs_distance = (
            calculate_improvement_percent(
                baseline_cost=(
                    mean_distance_cost
                ),

                candidate_cost=(
                    mean_ml_cost
                )
            )
        )

        # ------------------------------------------
        # Tail / P90 realized cost
        # ------------------------------------------

        distance_p90 = float(
            np.quantile(
                distance_costs,
                0.90
            )
        )

        free_flow_p90 = float(
            np.quantile(
                free_flow_costs,
                0.90
            )
        )

        ml_p90 = float(
            np.quantile(
                ml_costs,
                0.90
            )
        )

        oracle_route_p90 = float(
            np.quantile(
                oracle_route_costs,
                0.90
            )
        )

        ml_p90_improvement_vs_free_flow = (
            calculate_improvement_percent(
                baseline_cost=(
                    free_flow_p90
                ),

                candidate_cost=(
                    ml_p90
                )
            )
        )

        # ------------------------------------------
        # Win probabilities
        # ------------------------------------------

        probability_ml_beats_free_flow = float(
            np.mean(
                ml_costs
                < free_flow_costs
            )
        )

        probability_ml_beats_distance = float(
            np.mean(
                ml_costs
                < distance_costs
            )
        )

        # ------------------------------------------
        # Scenario summary row
        # ------------------------------------------

        scenario_summary_rows.append(
            {
                "network_seed":
                    network_seed,

                "scenario":
                    row["scenario"],

                "mean_distance_realized_cost":
                    mean_distance_cost,

                "mean_free_flow_realized_cost":
                    mean_free_flow_cost,

                "mean_ml_realized_cost":
                    mean_ml_cost,

                "mean_expected_oracle_route_realized_cost":
                    mean_oracle_route_cost,

                "mean_ml_improvement_vs_distance_percent":
                    mean_cost_ml_improvement_vs_distance,

                "mean_ml_improvement_vs_free_flow_percent":
                    mean_cost_ml_improvement_vs_free_flow,

                "mean_draw_level_ml_improvement_vs_free_flow_percent":
                    float(
                        np.mean(
                            ml_vs_free_flow_improvements
                        )
                    ),

                "distance_realized_std":
                    float(
                        np.std(
                            distance_costs,
                            ddof=1
                        )
                    ),

                "free_flow_realized_std":
                    float(
                        np.std(
                            free_flow_costs,
                            ddof=1
                        )
                    ),

                "ml_realized_std":
                    float(
                        np.std(
                            ml_costs,
                            ddof=1
                        )
                    ),

                "distance_p90_realized_cost":
                    distance_p90,

                "free_flow_p90_realized_cost":
                    free_flow_p90,

                "ml_p90_realized_cost":
                    ml_p90,

                "expected_oracle_route_p90_realized_cost":
                    oracle_route_p90,

                "ml_p90_improvement_vs_free_flow_percent":
                    ml_p90_improvement_vs_free_flow,

                "probability_ml_beats_free_flow":
                    probability_ml_beats_free_flow,

                "probability_ml_beats_distance":
                    probability_ml_beats_distance
            }
        )

        # ==================================================
        # Progress report
        # ==================================================

        print(
            f"Mean realized ML vs Free Flow: "
            f"{mean_cost_ml_improvement_vs_free_flow:.2f}%"
        )

        print(
            f"P90 ML vs Free Flow: "
            f"{ml_p90_improvement_vs_free_flow:.2f}%"
        )

        print(
            f"P(ML beats Free Flow): "
            f"{100 * probability_ml_beats_free_flow:.1f}%"
        )

    # ==================================================
    # Save raw Monte Carlo results
    # ==================================================

    realization_df = pd.DataFrame(
        realization_rows
    )

    realization_df.to_csv(
        "data/stochastic_realization_results.csv",
        index=False
    )

    # ==================================================
    # Save scenario-level summary
    # ==================================================

    scenario_summary_df = (
        pd.DataFrame(
            scenario_summary_rows
        )
    )

    scenario_summary_df.to_csv(
        "data/stochastic_scenario_summary.csv",
        index=False
    )

    # ==================================================
    # Overall descriptive report
    # ==================================================

    print(
        "\n"
        "========================================"
    )

    print(
        "STOCHASTIC ROUTE EVALUATION COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"Decision scenarios: "
        f"{len(scenario_summary_df)}"
    )

    print(
        f"Realizations per scenario: "
        f"{NUM_REALIZATIONS}"
    )

    print(
        f"Total realized evaluations: "
        f"{len(realization_df)}"
    )

    print(
        "\nMean realized improvement "
        "vs Free Flow:"
    )

    print(
        f"{scenario_summary_df['mean_ml_improvement_vs_free_flow_percent'].mean():.3f}%"
    )

    print(
        "\nMean P90 improvement "
        "vs Free Flow:"
    )

    print(
        f"{scenario_summary_df['ml_p90_improvement_vs_free_flow_percent'].mean():.3f}%"
    )

    print(
        "\nMean probability "
        "ML beats Free Flow:"
    )

    print(
        f"{100 * scenario_summary_df['probability_ml_beats_free_flow'].mean():.2f}%"
    )

    print(
        "\nMean realized improvement "
        "vs Distance:"
    )

    print(
        f"{scenario_summary_df['mean_ml_improvement_vs_distance_percent'].mean():.3f}%"
    )

    print(
        "\nSaved:"
    )

    print(
        "data/stochastic_realization_results.csv"
    )

    print(
        "data/stochastic_scenario_summary.csv"
    )


if __name__ == "__main__":
    main()

    