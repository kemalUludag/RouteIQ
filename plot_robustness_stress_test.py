import pandas as pd
import matplotlib.pyplot as plt

from src.result_analysis import (
    cluster_bootstrap_mean_confidence_interval
)


STRESS_ORDER = [
    "mild",
    "moderate",
    "severe"
]


STRESS_LABELS = [
    "Mild",
    "Moderate",
    "Severe"
]


def get_cluster_ci(
    df,
    column
):

    mean_value = (
        df[
            column
        ]
        .mean()
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

    return (
        mean_value,
        lower,
        upper
    )


def main():

    df = pd.read_csv(
        "data/robustness_stress_scenario_summary.csv"
    )

    # ==================================================
    # FIGURE 1
    # Mean routing gain remains stable
    # ==================================================

    means = []
    lower_errors = []
    upper_errors = []

    for stress in STRESS_ORDER:

        subset = (
            df[
                df[
                    "stress_level"
                ]
                == stress
            ]
            .copy()
        )

        mean_value, lower, upper = (
            get_cluster_ci(
                subset,
                "mean_ml_improvement_vs_free_flow_percent"
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
        figsize=(7, 5)
    )

    plt.errorbar(
        STRESS_LABELS,
        means,
        yerr=[
            lower_errors,
            upper_errors
        ],
        marker="o",
        capsize=6
    )

    plt.axhline(
        y=0,
        linewidth=1
    )

    plt.ylabel(
        "Mean ML Improvement vs Free Flow (%)"
    )

    plt.title(
        "Average Routing Gain under Distribution Shift"
    )

    plt.tight_layout()

    plt.savefig(
        "data/robustness_mean_gain.png",
        dpi=300
    )

    plt.close()

    # ==================================================
    # FIGURE 2
    # Reliability deterioration
    # ==================================================

    win_probabilities = []
    loss_probabilities = []

    for stress in STRESS_ORDER:

        subset = (
            df[
                df[
                    "stress_level"
                ]
                == stress
            ]
        )

        win_probabilities.append(
            subset[
                "win_probability"
            ]
            .mean()
            * 100
        )

        loss_probabilities.append(
            subset[
                "loss_probability"
            ]
            .mean()
            * 100
        )

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        STRESS_LABELS,
        win_probabilities,
        marker="o",
        label="Win probability"
    )

    plt.plot(
        STRESS_LABELS,
        loss_probabilities,
        marker="o",
        label="Loss probability"
    )

    plt.ylabel(
        "Realization Probability (%)"
    )

    plt.title(
        "Execution Reliability under Increasing Stress"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "data/robustness_reliability.png",
        dpi=300
    )

    plt.close()

    # ==================================================
    # FIGURE 3
    # Execution variability
    # ==================================================

    std_means = []
    std_lower_errors = []
    std_upper_errors = []

    for stress in STRESS_ORDER:

        subset = (
            df[
                df[
                    "stress_level"
                ]
                == stress
            ]
            .copy()
        )

        mean_value, lower, upper = (
            get_cluster_ci(
                subset,
                "ml_realized_std"
            )
        )

        std_means.append(
            mean_value
        )

        std_lower_errors.append(
            mean_value
            - lower
        )

        std_upper_errors.append(
            upper
            - mean_value
        )

    plt.figure(
        figsize=(7, 5)
    )

    plt.errorbar(
        STRESS_LABELS,
        std_means,
        yerr=[
            std_lower_errors,
            std_upper_errors
        ],
        marker="o",
        capsize=6
    )

    plt.ylabel(
        "Mean Realized Route-Time Std. Dev. (minutes)"
    )

    plt.title(
        "Execution Variability under Distribution Shift"
    )

    plt.tight_layout()

    plt.savefig(
        "data/robustness_variability.png",
        dpi=300
    )

    plt.close()

    print(
        "Saved:"
    )

    print(
        "data/robustness_mean_gain.png"
    )

    print(
        "data/robustness_reliability.png"
    )

    print(
        "data/robustness_variability.png"
    )


if __name__ == "__main__":
    main()
    