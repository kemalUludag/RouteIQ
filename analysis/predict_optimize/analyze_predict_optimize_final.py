import pandas as pd

from routeiq.result_analysis import (
    cluster_bootstrap_mean_confidence_interval
)


BOOTSTRAP_SAMPLES = 10000

CONFIDENCE_LEVEL = 0.95

RANDOM_STATE = 42


def main():

    # ==========================================
    # Load final benchmark
    # ==========================================

    df = pd.read_csv(
        "data/predict_optimize_final_results.csv"
    )

    # ==========================================
    # Additional derived metrics
    # ==========================================

    # How many percentage points of oracle
    # regret does ML remove relative to
    # the free-flow baseline?
    #
    # Positive = ML is closer to oracle.
    # ==========================================

    df[
        "ml_regret_reduction_vs_free_flow_pp"
    ] = (
        df[
            "free_flow_regret_vs_oracle_percent"
        ]
        -
        df[
            "ml_regret_vs_oracle_percent"
        ]
    )

    # ------------------------------------------
    # Exact objective parity with oracle
    # ------------------------------------------

    oracle_tolerance = 1e-6

    df[
        "ml_matches_oracle"
    ] = (
        df[
            "ml_regret_vs_oracle_percent"
        ]
        .abs()
        <= oracle_tolerance
    )

    # ==========================================
    # Overall descriptive statistics
    # ==========================================

    num_networks = (
        df[
            "network_seed"
        ]
        .nunique()
    )

    num_scenarios = len(
        df
    )

    # ------------------------------------------
    # Pairwise outcomes
    # ------------------------------------------

    distance_wins = (
        df[
            "ml_outcome_vs_distance"
        ]
        .eq("win")
        .sum()
    )

    distance_ties = (
        df[
            "ml_outcome_vs_distance"
        ]
        .eq("tie")
        .sum()
    )

    distance_losses = (
        df[
            "ml_outcome_vs_distance"
        ]
        .eq("loss")
        .sum()
    )

    free_flow_wins = (
        df[
            "ml_outcome_vs_free_flow"
        ]
        .eq("win")
        .sum()
    )

    free_flow_ties = (
        df[
            "ml_outcome_vs_free_flow"
        ]
        .eq("tie")
        .sum()
    )

    free_flow_losses = (
        df[
            "ml_outcome_vs_free_flow"
        ]
        .eq("loss")
        .sum()
    )

    # ==========================================
    # Cluster-bootstrap inference
    # ==========================================

    bootstrap_metrics = {
        "ML improvement vs distance (%)":
            "ml_improvement_vs_distance_percent",

        "ML improvement vs free flow (%)":
            "ml_improvement_vs_free_flow_percent",

        "Distance regret vs oracle (%)":
            "distance_regret_vs_oracle_percent",

        "Free-flow regret vs oracle (%)":
            "free_flow_regret_vs_oracle_percent",

        "ML regret vs oracle (%)":
            "ml_regret_vs_oracle_percent",

        "ML regret reduction vs free flow (pp)":
            "ml_regret_reduction_vs_free_flow_pp"
    }

    bootstrap_rows = []

    for (
        metric_name,
        column_name
    ) in bootstrap_metrics.items():

        mean_value = (
            df[
                column_name
            ]
            .mean()
        )

        lower, upper = (
            cluster_bootstrap_mean_confidence_interval(
                df=df,

                cluster_column=(
                    "network_seed"
                ),

                value_column=(
                    column_name
                ),

                confidence_level=(
                    CONFIDENCE_LEVEL
                ),

                num_bootstrap_samples=(
                    BOOTSTRAP_SAMPLES
                ),

                seed=(
                    RANDOM_STATE
                )
            )
        )

        bootstrap_rows.append(
            {
                "metric":
                    metric_name,

                "mean":
                    mean_value,

                "ci_lower":
                    lower,

                "ci_upper":
                    upper
            }
        )

    bootstrap_summary = (
        pd.DataFrame(
            bootstrap_rows
        )
    )

    # ==========================================
    # Scenario-level breakdown
    # ==========================================

    scenario_summary = (
        df
        .groupby(
            "scenario"
        )
        .agg(
            networks=(
                "network_seed",
                "nunique"
            ),

            mean_edge_mae=(
                "edge_prediction_mae",
                "mean"
            ),

            mean_ml_improvement_vs_distance=(
                "ml_improvement_vs_distance_percent",
                "mean"
            ),

            mean_ml_improvement_vs_free_flow=(
                "ml_improvement_vs_free_flow_percent",
                "mean"
            ),

            median_ml_improvement_vs_free_flow=(
                "ml_improvement_vs_free_flow_percent",
                "median"
            ),

            mean_free_flow_regret=(
                "free_flow_regret_vs_oracle_percent",
                "mean"
            ),

            mean_ml_regret=(
                "ml_regret_vs_oracle_percent",
                "mean"
            ),

            max_ml_regret=(
                "ml_regret_vs_oracle_percent",
                "max"
            ),

            ml_vs_free_flow_wins=(
                "ml_outcome_vs_free_flow",
                lambda values:
                    (
                        values
                        == "win"
                    ).sum()
            ),

            ml_vs_free_flow_ties=(
                "ml_outcome_vs_free_flow",
                lambda values:
                    (
                        values
                        == "tie"
                    ).sum()
            ),

            ml_vs_free_flow_losses=(
                "ml_outcome_vs_free_flow",
                lambda values:
                    (
                        values
                        == "loss"
                    ).sum()
            ),

            oracle_matches=(
                "ml_matches_oracle",
                "sum"
            )
        )
        .reset_index()
    )

    # ==========================================
    # Network-level summary
    #
    # One row now represents one independent
    # physical network.
    # ==========================================

    network_summary = (
        df
        .groupby(
            "network_seed"
        )
        .agg(
            scenarios=(
                "scenario",
                "count"
            ),

            mean_edge_mae=(
                "edge_prediction_mae",
                "mean"
            ),

            mean_ml_improvement_vs_distance=(
                "ml_improvement_vs_distance_percent",
                "mean"
            ),

            mean_ml_improvement_vs_free_flow=(
                "ml_improvement_vs_free_flow_percent",
                "mean"
            ),

            mean_free_flow_regret=(
                "free_flow_regret_vs_oracle_percent",
                "mean"
            ),

            mean_ml_regret=(
                "ml_regret_vs_oracle_percent",
                "mean"
            ),

            max_ml_regret=(
                "ml_regret_vs_oracle_percent",
                "max"
            )
        )
        .reset_index()
    )

    # ==========================================
    # Distribution of ML improvement
    # vs strong free-flow baseline
    # ==========================================

    free_flow_quantiles = (
        df[
            "ml_improvement_vs_free_flow_percent"
        ]
        .quantile(
            [
                0.05,
                0.25,
                0.50,
                0.75,
                0.95
            ]
        )
    )

    # ==========================================
    # Prediction accuracy vs decision quality
    # ==========================================

    pearson_mae_regret = (
        df[
            [
                "edge_prediction_mae",
                "ml_regret_vs_oracle_percent"
            ]
        ]
        .corr(
            method="pearson"
        )
        .iloc[
            0,
            1
        ]
    )

    spearman_mae_regret = (
        df[
            [
                "edge_prediction_mae",
                "ml_regret_vs_oracle_percent"
            ]
        ]
        .corr(
            method="spearman"
        )
        .iloc[
            0,
            1
        ]
    )

    pearson_mae_free_flow_improvement = (
        df[
            [
                "edge_prediction_mae",
                "ml_improvement_vs_free_flow_percent"
            ]
        ]
        .corr(
            method="pearson"
        )
        .iloc[
            0,
            1
        ]
    )

    # ==========================================
    # Oracle parity
    # ==========================================

    oracle_matches = int(
        df[
            "ml_matches_oracle"
        ]
        .sum()
    )

    oracle_match_rate = (
        oracle_matches
        /
        num_scenarios
        * 100
    )

    # ==========================================
    # Save outputs
    # ==========================================

    bootstrap_summary.to_csv(
        "data/predict_optimize_cluster_bootstrap.csv",
        index=False
    )

    scenario_summary.to_csv(
        "data/predict_optimize_final_scenario_summary.csv",
        index=False
    )

    network_summary.to_csv(
        "data/predict_optimize_final_network_summary.csv",
        index=False
    )

    # ==========================================
    # Report
    # ==========================================

    print(
        "\n"
        "========================================"
    )

    print(
        "FINAL PREDICT → OPTIMIZE ANALYSIS"
    )

    print(
        "========================================"
    )

    print(
        f"Independent networks: "
        f"{num_networks}"
    )

    print(
        f"Decision scenarios: "
        f"{num_scenarios}"
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
        "\nOracle objective matches:"
    )

    print(
        f"{oracle_matches}/"
        f"{num_scenarios} "
        f"({oracle_match_rate:.2f}%)"
    )

    # ==========================================
    # Bootstrap report
    # ==========================================

    print(
        "\n"
        "========================================"
    )

    print(
        "NETWORK-CLUSTER BOOTSTRAP"
    )

    print(
        "========================================"
    )

    print(
        bootstrap_summary.to_string(
            index=False
        )
    )

    # ==========================================
    # Scenario report
    # ==========================================

    print(
        "\n"
        "========================================"
    )

    print(
        "SCENARIO BREAKDOWN"
    )

    print(
        "========================================"
    )

    print(
        scenario_summary.to_string(
            index=False
        )
    )

    # ==========================================
    # Network report
    # ==========================================

    print(
        "\n"
        "========================================"
    )

    print(
        "NETWORK-LEVEL SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        network_summary.to_string(
            index=False
        )
    )

    # ==========================================
    # Quantiles
    # ==========================================

    print(
        "\n"
        "========================================"
    )

    print(
        "ML IMPROVEMENT VS FREE FLOW QUANTILES"
    )

    print(
        "========================================"
    )

    for quantile, value in (
        free_flow_quantiles.items()
    ):

        print(
            f"Q{int(quantile * 100):02d}: "
            f"{value:.3f}%"
        )

    # ==========================================
    # Prediction / decision relationship
    # ==========================================

    print(
        "\n"
        "========================================"
    )

    print(
        "PREDICTION vs DECISION QUALITY"
    )

    print(
        "========================================"
    )

    print(
        "Pearson("
        "edge MAE, ML oracle regret): "
        f"{pearson_mae_regret:.4f}"
    )

    print(
        "Spearman("
        "edge MAE, ML oracle regret): "
        f"{spearman_mae_regret:.4f}"
    )

    print(
        "Pearson("
        "edge MAE, ML improvement "
        "vs free flow): "
        f"{pearson_mae_free_flow_improvement:.4f}"
    )

    print(
        "\nSaved:"
    )

    print(
        "data/predict_optimize_cluster_bootstrap.csv"
    )

    print(
        "data/predict_optimize_final_scenario_summary.csv"
    )

    print(
        "data/predict_optimize_final_network_summary.csv"
    )


if __name__ == "__main__":
    main()

    