import pandas as pd
import matplotlib.pyplot as plt

from src.result_analysis import (
    cluster_bootstrap_mean_confidence_interval
)


def main():

    df = pd.read_csv(
        "data/predict_optimize_final_results.csv"
    )

    # ==================================================
    # FIGURE 1
    # Oracle regret comparison
    # ==================================================

    strategies = [
        (
            "Distance",
            "distance_regret_vs_oracle_percent"
        ),
        (
            "Free Flow",
            "free_flow_regret_vs_oracle_percent"
        ),
        (
            "ML Dynamic",
            "ml_regret_vs_oracle_percent"
        )
    ]

    labels = []
    means = []
    lower_errors = []
    upper_errors = []

    for label, column in strategies:

        mean_value = (
            df[column].mean()
        )

        lower, upper = (
            cluster_bootstrap_mean_confidence_interval(
                df=df,
                cluster_column="network_seed",
                value_column=column,
                confidence_level=0.95,
                num_bootstrap_samples=10000,
                seed=42
            )
        )

        labels.append(
            label
        )

        means.append(
            mean_value
        )

        lower_errors.append(
            mean_value
            - lower
        )

        upper_errors.append(
            upper
            - mean_value
        )

    plt.figure(
        figsize=(8, 5)
    )

    plt.errorbar(
        labels,
        means,
        yerr=[
            lower_errors,
            upper_errors
        ],
        fmt="o",
        capsize=6
    )

    plt.ylabel(
        "Mean Regret vs Exact Oracle (%)"
    )

    plt.title(
        "Routing Decision Quality vs Exact Oracle"
    )

    plt.tight_layout()

    plt.savefig(
        "data/predict_optimize_oracle_regret.png",
        dpi=300
    )

    plt.close()

    # ==================================================
    # FIGURE 2
    # ML improvement over strong free-flow baseline
    # by traffic regime
    # ==================================================

    scenario_order = [
        "weekday_midday",
        "weekday_rush",
        "weekend_midday",
        "weekend_evening"
    ]

    scenario_labels = [
        "Weekday\nMidday",
        "Weekday\nRush",
        "Weekend\nMidday",
        "Weekend\nEvening"
    ]

    scenario_means = []
    scenario_lower_errors = []
    scenario_upper_errors = []

    for scenario in scenario_order:

        subset = (
            df[
                df["scenario"]
                == scenario
            ]
            .copy()
        )

        mean_value = (
            subset[
                "ml_improvement_vs_free_flow_percent"
            ]
            .mean()
        )

        lower, upper = (
            cluster_bootstrap_mean_confidence_interval(
                df=subset,
                cluster_column="network_seed",
                value_column=(
                    "ml_improvement_vs_free_flow_percent"
                ),
                confidence_level=0.95,
                num_bootstrap_samples=10000,
                seed=42
            )
        )

        scenario_means.append(
            mean_value
        )

        scenario_lower_errors.append(
            mean_value
            - lower
        )

        scenario_upper_errors.append(
            upper
            - mean_value
        )

    plt.figure(
        figsize=(8, 5)
    )

    plt.errorbar(
        scenario_labels,
        scenario_means,
        yerr=[
            scenario_lower_errors,
            scenario_upper_errors
        ],
        fmt="o",
        capsize=6
    )

    plt.axhline(
        y=0,
        linewidth=1
    )

    plt.ylabel(
        "ML Improvement vs Free Flow (%)"
    )

    plt.title(
        "Value of Dynamic Prediction by Traffic Regime"
    )

    plt.tight_layout()

    plt.savefig(
        "data/predict_optimize_scenario_improvement.png",
        dpi=300
    )

    plt.close()

    # ==================================================
    # FIGURE 3
    # Full distribution of improvement over free flow
    # ==================================================

    scenario_values = [
        df.loc[
            df["scenario"]
            == scenario,
            "ml_improvement_vs_free_flow_percent"
        ].to_numpy()

        for scenario
        in scenario_order
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.boxplot(
        scenario_values,
        tick_labels=(
            scenario_labels
        ),
        showmeans=True
    )

    plt.axhline(
        y=0,
        linewidth=1
    )

    plt.ylabel(
        "ML Improvement vs Free Flow (%)"
    )

    plt.title(
        "Distribution of Dynamic Routing Gains"
    )

    plt.tight_layout()

    plt.savefig(
        "data/predict_optimize_improvement_distribution.png",
        dpi=300
    )

    plt.close()

    print(
        "Saved:"
    )

    print(
        "data/predict_optimize_oracle_regret.png"
    )

    print(
        "data/predict_optimize_scenario_improvement.png"
    )

    print(
        "data/predict_optimize_improvement_distribution.png"
    )


if __name__ == "__main__":
    main()

    