import pandas as pd
import matplotlib.pyplot as plt

from src.result_analysis import (
    cluster_bootstrap_mean_confidence_interval
)


def main():

    scenario_df = pd.read_csv(
        "data/stochastic_scenario_analysis.csv"
    )

    # ==================================================
    # Figure 1:
    # Mean realized improvement by regime
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

    means = []
    lower_errors = []
    upper_errors = []

    for scenario in scenario_order:

        subset = (
            scenario_df[
                scenario_df["scenario"]
                == scenario
            ]
            .copy()
        )

        mean_value = (
            subset[
                "mean_ml_improvement_vs_free_flow_percent"
            ]
            .mean()
        )

        lower, upper = (
            cluster_bootstrap_mean_confidence_interval(
                df=subset,
                cluster_column="network_seed",
                value_column=(
                    "mean_ml_improvement_vs_free_flow_percent"
                ),
                confidence_level=0.95,
                num_bootstrap_samples=10000,
                seed=42
            )
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
        scenario_labels,
        means,
        yerr=[
            lower_errors,
            upper_errors
        ],
        marker="o",
        capsize=6
    )

    plt.axhline(
        0,
        linewidth=1
    )

    plt.ylabel(
        "Mean Realized ML Improvement vs Free Flow (%)"
    )

    plt.title(
        "Stochastic Routing Gains by Traffic Regime"
    )

    plt.tight_layout()

    plt.savefig(
        "data/stochastic_regime_improvement.png",
        dpi=300
    )

    plt.close()

    # ==================================================
    # Figure 2:
    # Expected vs realized improvement
    # ==================================================

    plt.figure(
        figsize=(7, 6)
    )

    x = (
        scenario_df[
            "expected_ml_improvement_vs_free_flow_percent"
        ]
    )

    y = (
        scenario_df[
            "mean_ml_improvement_vs_free_flow_percent"
        ]
    )

    plt.scatter(
        x,
        y,
        alpha=0.7
    )

    minimum = min(
        x.min(),
        y.min()
    )

    maximum = max(
        x.max(),
        y.max()
    )

    plt.plot(
        [
            minimum,
            maximum
        ],
        [
            minimum,
            maximum
        ],
        linestyle="--"
    )

    plt.xlabel(
        "Expected-Cost Improvement vs Free Flow (%)"
    )

    plt.ylabel(
        "Mean Realized Improvement vs Free Flow (%)"
    )

    plt.title(
        "Expected vs Realized Routing Gains"
    )

    plt.tight_layout()

    plt.savefig(
        "data/stochastic_expected_vs_realized.png",
        dpi=300
    )

    plt.close()

    # ==================================================
    # Figure 3:
    # Win / tie / loss probabilities by regime
    # ==================================================

    regime_summary = (
        scenario_df
        .groupby(
            "scenario"
        )
        .agg(
            win_probability=(
                "win_probability",
                "mean"
            ),

            tie_probability=(
                "tie_probability",
                "mean"
            ),

            loss_probability=(
                "loss_probability",
                "mean"
            )
        )
        .reindex(
            scenario_order
        )
    )

    x_positions = range(
        len(
            scenario_order
        )
    )

    width = 0.25

    plt.figure(
        figsize=(9, 5)
    )

    plt.bar(
        [
            position - width
            for position in x_positions
        ],
        regime_summary[
            "win_probability"
        ]
        * 100,
        width=width,
        label="Win"
    )

    plt.bar(
        x_positions,
        regime_summary[
            "tie_probability"
        ]
        * 100,
        width=width,
        label="Tie"
    )

    plt.bar(
        [
            position + width
            for position in x_positions
        ],
        regime_summary[
            "loss_probability"
        ]
        * 100,
        width=width,
        label="Loss"
    )

    plt.xticks(
        list(
            x_positions
        ),
        scenario_labels
    )

    plt.ylabel(
        "Mean Realization Probability (%)"
    )

    plt.title(
        "ML vs Free-Flow Outcomes under Stochastic Execution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "data/stochastic_outcome_probabilities.png",
        dpi=300
    )

    plt.close()

    print(
        "Saved:"
    )

    print(
        "data/stochastic_regime_improvement.png"
    )

    print(
        "data/stochastic_expected_vs_realized.png"
    )

    print(
        "data/stochastic_outcome_probabilities.png"
    )


if __name__ == "__main__":
    main()


    