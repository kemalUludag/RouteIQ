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

from routeiq.robustness_simulation import (
    STRESS_CONFIGS,
    build_stress_realization
)

from routeiq.travel_time_matrix import (
    generate_routing_context
)


# ==================================================
# Experiment configuration
# ==================================================

NUM_CUSTOMERS = 10

NUM_VEHICLES = 2

VEHICLE_CAPACITY = 15

NUM_REALIZATIONS = 100

TOLERANCE = 1e-9


def classify_outcome(
    candidate_cost,
    baseline_cost
):
    difference = (
        baseline_cost
        - candidate_cost
    )

    if difference > TOLERANCE:
        return "win"

    if difference < -TOLERANCE:
        return "loss"

    return "tie"


def main():

    # ==================================================
    # Load FINALIZED routing decisions
    #
    # These routes were already selected before
    # robustness testing.
    #
    # No reoptimization occurs in this script.
    # ==================================================

    decision_df = pd.read_csv(
        "results/predict_optimize/predict_optimize_final_results.csv"
    )

    realization_rows = []

    scenario_summary_rows = []

    # ==================================================
    # Decision-scenario loop
    # ==================================================

    for row_index, row in (
        decision_df.iterrows()
    ):

        network_seed = int(
            row["network_seed"]
        )

        scenario_name = str(
            row["scenario"]
        )

        print(
            "\n"
            "========================================"
        )

        print(
            f"DECISION SCENARIO "
            f"{row_index + 1}/"
            f"{len(decision_df)}"
        )

        print(
            f"Network:  {network_seed}"
        )

        print(
            f"Scenario: {scenario_name}"
        )

        print(
            "========================================"
        )

        # ==================================================
        # Reconstruct physical CVRP network
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
        # Recover locked route decisions
        # ==================================================

        distance_routes = (
            json.loads(
                row[
                    "distance_routes"
                ]
            )
        )

        free_flow_routes = (
            json.loads(
                row[
                    "free_flow_routes"
                ]
            )
        )

        ml_routes = (
            json.loads(
                row[
                    "ml_routes"
                ]
            )
        )

        expected_oracle_routes = (
            json.loads(
                row[
                    "oracle_routes"
                ]
            )
        )

        # ==================================================
        # Stress-level loop
        # ==================================================

        for (
            stress_name,
            stress_config
        ) in STRESS_CONFIGS.items():

            print(
                f"\nStress level: "
                f"{stress_name}"
            )

            distance_costs = []

            free_flow_costs = []

            ml_costs = []

            expected_oracle_costs = []

            ml_vs_free_flow_improvements = []

            ml_vs_distance_improvements = []

            ml_outcomes_vs_free_flow = []

            common_shocks = []

            # ==================================================
            # Monte Carlo loop
            # ==================================================

            for realization_index in range(
                NUM_REALIZATIONS
            ):

                # ------------------------------------------
                # IMPORTANT:
                #
                # The same realization seed is reused
                # across mild / moderate / severe.
                #
                # This creates a paired stress comparison.
                # ------------------------------------------

                realization_seed = (
                    2_000_000
                    + row_index * 10_000
                    + realization_index
                )

                stress_realization = (
                    build_stress_realization(
                        instance=instance,

                        scenario=scenario,

                        seed=(
                            realization_seed
                        ),

                        config=(
                            stress_config
                        )
                    )
                )

                realized_matrix = (
                    stress_realization
                    .travel_time_matrix
                )

                common_shock = (
                    stress_realization
                    .common_shock
                )

                # ==================================================
                # Fair route evaluation
                #
                # SAME realized matrix is used for every
                # routing strategy.
                # ==================================================

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

                expected_oracle_cost = (
                    evaluate_routes_on_matrix(
                        expected_oracle_routes,
                        realized_matrix
                    )
                )

                # ==================================================
                # Pairwise improvements
                # ==================================================

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

                outcome_vs_free_flow = (
                    classify_outcome(
                        candidate_cost=(
                            ml_cost
                        ),

                        baseline_cost=(
                            free_flow_cost
                        )
                    )
                )

                # ==================================================
                # Local arrays
                # ==================================================

                distance_costs.append(
                    distance_cost
                )

                free_flow_costs.append(
                    free_flow_cost
                )

                ml_costs.append(
                    ml_cost
                )

                expected_oracle_costs.append(
                    expected_oracle_cost
                )

                ml_vs_free_flow_improvements.append(
                    ml_vs_free_flow
                )

                ml_vs_distance_improvements.append(
                    ml_vs_distance
                )

                ml_outcomes_vs_free_flow.append(
                    outcome_vs_free_flow
                )

                common_shocks.append(
                    common_shock
                )

                # ==================================================
                # Raw realization record
                # ==================================================

                realization_rows.append(
                    {
                        "network_seed":
                            network_seed,

                        "scenario":
                            scenario_name,

                        "decision_scenario_index":
                            row_index,

                        "stress_level":
                            stress_name,

                        "realization_index":
                            realization_index,

                        "realization_seed":
                            realization_seed,

                        "common_shock":
                            common_shock,

                        "distance_realized_cost":
                            distance_cost,

                        "free_flow_realized_cost":
                            free_flow_cost,

                        "ml_realized_cost":
                            ml_cost,

                        "expected_oracle_route_realized_cost":
                            expected_oracle_cost,

                        "ml_improvement_vs_distance_percent":
                            ml_vs_distance,

                        "ml_improvement_vs_free_flow_percent":
                            ml_vs_free_flow,

                        "ml_outcome_vs_free_flow":
                            outcome_vs_free_flow
                    }
                )

            # ==================================================
            # Convert arrays
            # ==================================================

            distance_costs = np.asarray(
                distance_costs,
                dtype=float
            )

            free_flow_costs = np.asarray(
                free_flow_costs,
                dtype=float
            )

            ml_costs = np.asarray(
                ml_costs,
                dtype=float
            )

            expected_oracle_costs = (
                np.asarray(
                    expected_oracle_costs,
                    dtype=float
                )
            )

            ml_vs_free_flow_improvements = (
                np.asarray(
                    ml_vs_free_flow_improvements,
                    dtype=float
                )
            )

            ml_vs_distance_improvements = (
                np.asarray(
                    ml_vs_distance_improvements,
                    dtype=float
                )
            )

            common_shocks = np.asarray(
                common_shocks,
                dtype=float
            )

            # ==================================================
            # Mean realized costs
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

            mean_expected_oracle_cost = (
                float(
                    np.mean(
                        expected_oracle_costs
                    )
                )
            )

            # ==================================================
            # Improvement based on mean costs
            #
            # This is our primary mean-performance metric.
            # ==================================================

            mean_ml_improvement_vs_free_flow = (
                calculate_improvement_percent(
                    baseline_cost=(
                        mean_free_flow_cost
                    ),

                    candidate_cost=(
                        mean_ml_cost
                    )
                )
            )

            mean_ml_improvement_vs_distance = (
                calculate_improvement_percent(
                    baseline_cost=(
                        mean_distance_cost
                    ),

                    candidate_cost=(
                        mean_ml_cost
                    )
                )
            )

            # ==================================================
            # P90 realized costs
            # ==================================================

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

            expected_oracle_p90 = float(
                np.quantile(
                    expected_oracle_costs,
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

            # ==================================================
            # Win / tie / loss probabilities
            # ==================================================

            outcomes = np.asarray(
                ml_outcomes_vs_free_flow
            )

            win_probability = float(
                np.mean(
                    outcomes
                    == "win"
                )
            )

            tie_probability = float(
                np.mean(
                    outcomes
                    == "tie"
                )
            )

            loss_probability = float(
                np.mean(
                    outcomes
                    == "loss"
                )
            )

            non_tie_probability = (
                win_probability
                +
                loss_probability
            )

            if (
                non_tie_probability
                > TOLERANCE
            ):

                conditional_win_probability = (
                    win_probability
                    /
                    non_tie_probability
                )

            else:

                conditional_win_probability = (
                    float("nan")
                )

            # ==================================================
            # Scenario-level stress summary
            # ==================================================

            scenario_summary_rows.append(
                {
                    "network_seed":
                        network_seed,

                    "scenario":
                        scenario_name,

                    "decision_scenario_index":
                        row_index,

                    "stress_level":
                        stress_name,

                    # ----------------------------------
                    # Original expected decision quality
                    # ----------------------------------

                    "expected_ml_improvement_vs_free_flow_percent":
                        float(
                            row[
                                "ml_improvement_vs_free_flow_percent"
                            ]
                        ),

                    "expected_ml_regret_vs_oracle_percent":
                        float(
                            row[
                                "ml_regret_vs_oracle_percent"
                            ]
                        ),

                    # ----------------------------------
                    # Mean realized costs
                    # ----------------------------------

                    "mean_distance_realized_cost":
                        mean_distance_cost,

                    "mean_free_flow_realized_cost":
                        mean_free_flow_cost,

                    "mean_ml_realized_cost":
                        mean_ml_cost,

                    "mean_expected_oracle_route_realized_cost":
                        mean_expected_oracle_cost,

                    # ----------------------------------
                    # Mean improvements
                    # ----------------------------------

                    "mean_ml_improvement_vs_distance_percent":
                        mean_ml_improvement_vs_distance,

                    "mean_ml_improvement_vs_free_flow_percent":
                        mean_ml_improvement_vs_free_flow,

                    "mean_draw_level_ml_improvement_vs_free_flow_percent":
                        float(
                            np.mean(
                                ml_vs_free_flow_improvements
                            )
                        ),

                    # ----------------------------------
                    # Variability
                    # ----------------------------------

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

                    # ----------------------------------
                    # Tail performance
                    # ----------------------------------

                    "distance_p90_realized_cost":
                        distance_p90,

                    "free_flow_p90_realized_cost":
                        free_flow_p90,

                    "ml_p90_realized_cost":
                        ml_p90,

                    "expected_oracle_route_p90_realized_cost":
                        expected_oracle_p90,

                    "ml_p90_improvement_vs_free_flow_percent":
                        ml_p90_improvement_vs_free_flow,

                    # ----------------------------------
                    # Outcome probabilities
                    # ----------------------------------

                    "win_probability":
                        win_probability,

                    "tie_probability":
                        tie_probability,

                    "loss_probability":
                        loss_probability,

                    "conditional_win_probability":
                        conditional_win_probability,

                    # ----------------------------------
                    # Shock diagnostics
                    # ----------------------------------

                    "mean_common_shock":
                        float(
                            np.mean(
                                common_shocks
                            )
                        ),

                    "common_shock_std_observed":
                        float(
                            np.std(
                                common_shocks,
                                ddof=1
                            )
                        )
                }
            )

            # ==================================================
            # Progress report
            # ==================================================

            print(
                f"Mean ML vs Free Flow: "
                f"{mean_ml_improvement_vs_free_flow:.2f}%"
            )

            print(
                f"P90 ML vs Free Flow: "
                f"{ml_p90_improvement_vs_free_flow:.2f}%"
            )

            print(
                f"P(win): "
                f"{100 * win_probability:.1f}%"
            )

            print(
                f"P(loss): "
                f"{100 * loss_probability:.1f}%"
            )

    # ==================================================
    # Save realization-level output
    # ==================================================

    realization_df = pd.DataFrame(
        realization_rows
    )

    realization_df.to_csv(
        "results/robustness/robustness_stress_realizations.csv",
        index=False
    )

    # ==================================================
    # Save scenario × stress summary
    #
    # 120 scenarios × 3 stress levels
    # = 360 rows
    # ==================================================

    scenario_summary_df = pd.DataFrame(
        scenario_summary_rows
    )

    scenario_summary_df.to_csv(
        "results/robustness/robustness_stress_scenario_summary.csv",
        index=False
    )

    # ==================================================
    # Overall stress-level descriptive summary
    # ==================================================

    stress_summary = (
        scenario_summary_df
        .groupby(
            "stress_level"
        )
        .agg(
            scenarios=(
                "decision_scenario_index",
                "count"
            ),

            mean_ml_improvement_vs_free_flow=(
                "mean_ml_improvement_vs_free_flow_percent",
                "mean"
            ),

            mean_ml_p90_improvement_vs_free_flow=(
                "ml_p90_improvement_vs_free_flow_percent",
                "mean"
            ),

            mean_ml_improvement_vs_distance=(
                "mean_ml_improvement_vs_distance_percent",
                "mean"
            ),

            mean_win_probability=(
                "win_probability",
                "mean"
            ),

            mean_tie_probability=(
                "tie_probability",
                "mean"
            ),

            mean_loss_probability=(
                "loss_probability",
                "mean"
            ),

            mean_conditional_win_probability=(
                "conditional_win_probability",
                "mean"
            ),

            mean_ml_realized_std=(
                "ml_realized_std",
                "mean"
            )
        )
        .reset_index()
    )

    # Keep logical stress order

    stress_order = {
        "mild": 0,
        "moderate": 1,
        "severe": 2
    }

    stress_summary[
        "_order"
    ] = (
        stress_summary[
            "stress_level"
        ]
        .map(
            stress_order
        )
    )

    stress_summary = (
        stress_summary
        .sort_values(
            "_order"
        )
        .drop(
            columns="_order"
        )
        .reset_index(
            drop=True
        )
    )

    stress_summary.to_csv(
        "results/robustness/robustness_stress_summary.csv",
        index=False
    )

    # ==================================================
    # Final report
    # ==================================================

    print(
        "\n"
        "========================================"
    )

    print(
        "ROBUSTNESS STRESS TEST COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"Independent networks: "
        f"{scenario_summary_df['network_seed'].nunique()}"
    )

    print(
        f"Decision scenarios: "
        f"{decision_df.shape[0]}"
    )

    print(
        f"Stress levels: "
        f"{len(STRESS_CONFIGS)}"
    )

    print(
        f"Realizations per stress/scenario: "
        f"{NUM_REALIZATIONS}"
    )

    print(
        f"Total stress realizations: "
        f"{len(realization_df)}"
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "STRESS-LEVEL SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        stress_summary.to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )

    print(
        "results/robustness/robustness_stress_realizations.csv"
    )

    print(
        "results/robustness/robustness_stress_scenario_summary.csv"
    )

    print(
        "results/robustness/robustness_stress_summary.csv"
    )


if __name__ == "__main__":
    main()


    