import numpy as np
import pandas as pd

from src.result_analysis import (
    cluster_bootstrap_mean_confidence_interval
)


BOOTSTRAP_SAMPLES = 10000

CONFIDENCE_LEVEL = 0.95

RANDOM_STATE = 42

TOLERANCE = 1e-9


def main():

    # ==================================================
    # Load stochastic experiment outputs
    # ==================================================

    realization_df = pd.read_csv(
        "data/stochastic_realization_results.csv"
    )

    scenario_df = pd.read_csv(
        "data/stochastic_scenario_summary.csv"
    )

    decision_df = pd.read_csv(
        "data/predict_optimize_final_results.csv"
    )

    # ==================================================
    # 1. Realization-level paired cost difference
    #
    # Positive:
    # ML is cheaper / better.
    #
    # Negative:
    # Free Flow is cheaper / better.
    # ==================================================

    realization_df[
        "ml_savings_vs_free_flow_minutes"
    ] = (
        realization_df[
            "free_flow_realized_cost"
        ]
        -
        realization_df[
            "ml_realized_cost"
        ]
    )

    # ==================================================
    # 2. Tie-aware realization outcomes
    # ==================================================

    savings = (
        realization_df[
            "ml_savings_vs_free_flow_minutes"
        ]
    )

    realization_df[
        "ml_vs_free_flow_outcome"
    ] = np.where(
        savings > TOLERANCE,
        "win",
        np.where(
            savings < -TOLERANCE,
            "loss",
            "tie"
        )
    )

    # ==================================================
    # 3. Scenario-level win / tie / loss probabilities
    # ==================================================

    outcome_summary = (
        realization_df
        .groupby(
            [
                "network_seed",
                "scenario",
                "decision_scenario_index"
            ]
        )
        .agg(
            realization_count=(
                "realization_index",
                "count"
            ),

            win_probability=(
                "ml_vs_free_flow_outcome",
                lambda values:
                    np.mean(
                        values == "win"
                    )
            ),

            tie_probability=(
                "ml_vs_free_flow_outcome",
                lambda values:
                    np.mean(
                        values == "tie"
                    )
            ),

            loss_probability=(
                "ml_vs_free_flow_outcome",
                lambda values:
                    np.mean(
                        values == "loss"
                    )
            ),

            mean_savings_minutes=(
                "ml_savings_vs_free_flow_minutes",
                "mean"
            ),

            median_savings_minutes=(
                "ml_savings_vs_free_flow_minutes",
                "median"
            )
        )
        .reset_index()
    )

    # ==================================================
    # 4. Probability of winning conditional on
    # a non-tied realization
    #
    # If all 100 realizations are ties because the
    # exact same route was selected, this metric
    # remains NaN rather than pretending it is 0%.
    # ==================================================

    non_tie_probability = (
        outcome_summary[
            "win_probability"
        ]
        +
        outcome_summary[
            "loss_probability"
        ]
    )

    outcome_summary[
        "conditional_win_probability"
    ] = np.where(
        non_tie_probability
        > TOLERANCE,

        outcome_summary[
            "win_probability"
        ]
        /
        non_tie_probability,

        np.nan
    )

    # ==================================================
    # 5. Merge with stochastic scenario metrics
    # ==================================================

    scenario_analysis = (
        scenario_df.merge(
            outcome_summary,
            on=[
                "network_seed",
                "scenario"
            ],
            how="left"
        )
    )

    # ==================================================
    # 6. Bring expected-cost decision outcome
    # into the stochastic analysis
    # ==================================================

    decision_columns = (
        decision_df[
            [
                "network_seed",
                "scenario",
                "ml_improvement_vs_free_flow_percent",
                "ml_outcome_vs_free_flow",
                "ml_regret_vs_oracle_percent"
            ]
        ]
        .rename(
            columns={
                "ml_improvement_vs_free_flow_percent":
                    "expected_ml_improvement_vs_free_flow_percent",

                "ml_outcome_vs_free_flow":
                    "expected_outcome_vs_free_flow",

                "ml_regret_vs_oracle_percent":
                    "expected_ml_regret_vs_oracle_percent"
            }
        )
    )

    scenario_analysis = (
        scenario_analysis.merge(
            decision_columns,
            on=[
                "network_seed",
                "scenario"
            ],
            how="left"
        )
    )

    # ==================================================
    # 7. Network-cluster bootstrap
    #
    # Again:
    #
    # 120 scenarios are NOT treated as
    # 120 independent observations.
    #
    # 30 networks are the resampling clusters.
    # ==================================================

    bootstrap_metrics = {
        "Mean realized ML improvement vs Free Flow (%)":
            (
                "mean_ml_improvement_vs_free_flow_percent"
            ),

        "P90 ML improvement vs Free Flow (%)":
            (
                "ml_p90_improvement_vs_free_flow_percent"
            ),

        "Mean realized ML improvement vs Distance (%)":
            (
                "mean_ml_improvement_vs_distance_percent"
            ),

        "ML win probability":
            (
                "win_probability"
            ),

        "ML tie probability":
            (
                "tie_probability"
            ),

        "ML loss probability":
            (
                "loss_probability"
            ),

        "Mean savings vs Free Flow (minutes)":
            (
                "mean_savings_minutes"
            )
    }

    bootstrap_rows = []

    for metric_name, column_name in (
        bootstrap_metrics.items()
    ):

        mean_value = (
            scenario_analysis[
                column_name
            ]
            .mean()
        )

        lower, upper = (
            cluster_bootstrap_mean_confidence_interval(
                df=scenario_analysis,

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

    bootstrap_df = pd.DataFrame(
        bootstrap_rows
    )

    # ==================================================
    # 8. Scenario-regime summary
    # ==================================================

    regime_summary = (
        scenario_analysis
        .groupby(
            "scenario"
        )
        .agg(
            networks=(
                "network_seed",
                "nunique"
            ),

            mean_realized_improvement=(
                "mean_ml_improvement_vs_free_flow_percent",
                "mean"
            ),

            mean_p90_improvement=(
                "ml_p90_improvement_vs_free_flow_percent",
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

            mean_savings_minutes=(
                "mean_savings_minutes",
                "mean"
            )
        )
        .reset_index()
    )

    # ==================================================
    # 9. Compare stochastic performance by
    # expected-cost decision outcome
    #
    # Example:
    #
    # If expected analysis said "ML win",
    # how often does ML still win after noise?
    # ==================================================

    expected_outcome_summary = (
        scenario_analysis
        .groupby(
            "expected_outcome_vs_free_flow"
        )
        .agg(
            scenarios=(
                "network_seed",
                "count"
            ),

            mean_realized_improvement=(
                "mean_ml_improvement_vs_free_flow_percent",
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
            )
        )
        .reset_index()
    )

    # ==================================================
    # 10. Overall realization counts
    #
    # Descriptive only.
    #
    # We do NOT use 12,000 as an independent
    # statistical sample size.
    # ==================================================

    total_wins = int(
        (
            realization_df[
                "ml_vs_free_flow_outcome"
            ]
            == "win"
        ).sum()
    )

    total_ties = int(
        (
            realization_df[
                "ml_vs_free_flow_outcome"
            ]
            == "tie"
        ).sum()
    )

    total_losses = int(
        (
            realization_df[
                "ml_vs_free_flow_outcome"
            ]
            == "loss"
        ).sum()
    )

    total_non_ties = (
        total_wins
        +
        total_losses
    )

    if total_non_ties > 0:

        conditional_win_rate = (
            total_wins
            /
            total_non_ties
        )

    else:

        conditional_win_rate = (
            float("nan")
        )

    # ==================================================
    # 11. Expected vs realized improvement relationship
    # ==================================================

    expected_realized_pearson = (
        scenario_analysis[
            [
                "expected_ml_improvement_vs_free_flow_percent",
                "mean_ml_improvement_vs_free_flow_percent"
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

    expected_realized_spearman = (
        scenario_analysis[
            [
                "expected_ml_improvement_vs_free_flow_percent",
                "mean_ml_improvement_vs_free_flow_percent"
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

    # ==================================================
    # 12. Save outputs
    # ==================================================

    realization_df.to_csv(
        "data/stochastic_realization_analysis.csv",
        index=False
    )

    scenario_analysis.to_csv(
        "data/stochastic_scenario_analysis.csv",
        index=False
    )

    bootstrap_df.to_csv(
        "data/stochastic_cluster_bootstrap.csv",
        index=False
    )

    regime_summary.to_csv(
        "data/stochastic_regime_summary.csv",
        index=False
    )

    expected_outcome_summary.to_csv(
        "data/stochastic_expected_outcome_summary.csv",
        index=False
    )

    # ==================================================
    # 13. Report
    # ==================================================

    print(
        "\n"
        "========================================"
    )

    print(
        "STOCHASTIC ROUTING ANALYSIS"
    )

    print(
        "========================================"
    )

    print(
        f"Independent networks: "
        f"{scenario_analysis['network_seed'].nunique()}"
    )

    print(
        f"Decision scenarios: "
        f"{len(scenario_analysis)}"
    )

    print(
        f"Realizations: "
        f"{len(realization_df)}"
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "REALIZATION OUTCOMES — DESCRIPTIVE"
    )

    print(
        "========================================"
    )

    print(
        f"ML wins:   "
        f"{total_wins}"
    )

    print(
        f"Ties:      "
        f"{total_ties}"
    )

    print(
        f"ML losses: "
        f"{total_losses}"
    )

    print(
        f"\nP(win):  "
        f"{100 * total_wins / len(realization_df):.2f}%"
    )

    print(
        f"P(tie):   "
        f"{100 * total_ties / len(realization_df):.2f}%"
    )

    print(
        f"P(loss):  "
        f"{100 * total_losses / len(realization_df):.2f}%"
    )

    print(
        "\nConditional on a non-tied realization:"
    )

    print(
        f"P(ML wins | non-tie): "
        f"{100 * conditional_win_rate:.2f}%"
    )

    # ==================================================
    # Bootstrap
    # ==================================================

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
        bootstrap_df.to_string(
            index=False
        )
    )

    # ==================================================
    # Regimes
    # ==================================================

    print(
        "\n"
        "========================================"
    )

    print(
        "TRAFFIC-REGIME SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        regime_summary.to_string(
            index=False
        )
    )

    # ==================================================
    # Expected outcome vs stochastic outcome
    # ==================================================

    print(
        "\n"
        "========================================"
    )

    print(
        "EXPECTED DECISION vs REALIZED PERFORMANCE"
    )

    print(
        "========================================"
    )

    print(
        expected_outcome_summary.to_string(
            index=False
        )
    )

    # ==================================================
    # Expected / realized relationship
    # ==================================================

    print(
        "\n"
        "========================================"
    )

    print(
        "EXPECTED vs REALIZED IMPROVEMENT"
    )

    print(
        "========================================"
    )

    print(
        f"Pearson:  "
        f"{expected_realized_pearson:.4f}"
    )

    print(
        f"Spearman: "
        f"{expected_realized_spearman:.4f}"
    )

    print(
        "\nSaved:"
    )

    print(
        "data/stochastic_realization_analysis.csv"
    )

    print(
        "data/stochastic_scenario_analysis.csv"
    )

    print(
        "data/stochastic_cluster_bootstrap.csv"
    )

    print(
        "data/stochastic_regime_summary.csv"
    )

    print(
        "data/stochastic_expected_outcome_summary.csv"
    )


if __name__ == "__main__":
    main()


    