import numpy as np
import pandas as pd

from src.result_analysis import (
    cluster_bootstrap_mean_confidence_interval
)


BOOTSTRAP_SAMPLES = 10000

CONFIDENCE_LEVEL = 0.95

RANDOM_STATE = 42


STRESS_ORDER = [
    "mild",
    "moderate",
    "severe"
]


STRESS_COMPARISONS = [
    (
        "mild",
        "moderate"
    ),
    (
        "moderate",
        "severe"
    ),
    (
        "mild",
        "severe"
    )
]


# ==================================================
# Helper:
# cluster-bootstrap one metric
# ==================================================

def bootstrap_metric(
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

            cluster_column=(
                "network_seed"
            ),

            value_column=(
                column
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

    return (
        mean_value,
        lower,
        upper
    )


def main():

    # ==================================================
    # Load stress-test scenario summary
    #
    # 120 decision scenarios
    # ×
    # 3 stress levels
    #
    # = 360 rows
    # ==================================================

    df = pd.read_csv(
        "data/robustness_stress_scenario_summary.csv"
    )

    # ==================================================
    # Sanity checks
    # ==================================================

    expected_stress_levels = set(
        STRESS_ORDER
    )

    observed_stress_levels = set(
        df[
            "stress_level"
        ].unique()
    )

    if (
        observed_stress_levels
        != expected_stress_levels
    ):

        raise ValueError(
            "Unexpected stress levels. "
            f"Observed: "
            f"{observed_stress_levels}"
        )

    # ==================================================
    # Metrics to analyze at each stress level
    # ==================================================

    stress_metrics = {
        "Mean ML improvement vs Free Flow (%)":
            (
                "mean_ml_improvement_vs_free_flow_percent"
            ),

        "P90 ML improvement vs Free Flow (%)":
            (
                "ml_p90_improvement_vs_free_flow_percent"
            ),

        "ML win probability":
            (
                "win_probability"
            ),

        "ML loss probability":
            (
                "loss_probability"
            ),

        "Conditional ML win probability":
            (
                "conditional_win_probability"
            ),

        "ML realized standard deviation (min)":
            (
                "ml_realized_std"
            )
    }

    # ==================================================
    # 1. Stress-level cluster-bootstrap results
    # ==================================================

    stress_bootstrap_rows = []

    for stress_level in (
        STRESS_ORDER
    ):

        stress_df = (
            df[
                df[
                    "stress_level"
                ]
                == stress_level
            ]
            .copy()
        )

        for (
            metric_name,
            column_name
        ) in stress_metrics.items():

            usable_df = (
                stress_df[
                    [
                        "network_seed",
                        column_name
                    ]
                ]
                .dropna()
                .copy()
            )

            mean_value, lower, upper = (
                bootstrap_metric(
                    usable_df,
                    column_name
                )
            )

            stress_bootstrap_rows.append(
                {
                    "stress_level":
                        stress_level,

                    "metric":
                        metric_name,

                    "observations":
                        len(
                            usable_df
                        ),

                    "networks":
                        usable_df[
                            "network_seed"
                        ]
                        .nunique(),

                    "mean":
                        mean_value,

                    "ci_lower":
                        lower,

                    "ci_upper":
                        upper
                }
            )

    stress_bootstrap_df = (
        pd.DataFrame(
            stress_bootstrap_rows
        )
    )

    # ==================================================
    # 2. Paired stress deterioration / change
    #
    # First pivot:
    #
    # network + scenario
    #
    #        mild   moderate   severe
    #
    # Then compute:
    #
    # new stress - old stress
    #
    # This preserves scenario pairing.
    # ==================================================

    paired_metrics = {
        "Mean improvement change (pp)":
            (
                "mean_ml_improvement_vs_free_flow_percent"
            ),

        "P90 improvement change (pp)":
            (
                "ml_p90_improvement_vs_free_flow_percent"
            ),

        "Win probability change (pp)":
            (
                "win_probability"
            ),

        "Loss probability change (pp)":
            (
                "loss_probability"
            ),

        "Conditional win probability change (pp)":
            (
                "conditional_win_probability"
            ),

        "ML realized std change (min)":
            (
                "ml_realized_std"
            )
    }

    paired_change_rows = []

    for (
        old_stress,
        new_stress
    ) in STRESS_COMPARISONS:

        for (
            metric_name,
            column_name
        ) in paired_metrics.items():

            pivot_df = (
                df.pivot(
                    index=[
                        "network_seed",
                        "scenario"
                    ],

                    columns=(
                        "stress_level"
                    ),

                    values=(
                        column_name
                    )
                )
                .reset_index()
            )

            paired_df = (
                pivot_df[
                    [
                        "network_seed",
                        "scenario",
                        old_stress,
                        new_stress
                    ]
                ]
                .dropna()
                .copy()
            )

            difference_column = (
                "paired_change"
            )

            paired_df[
                difference_column
            ] = (
                paired_df[
                    new_stress
                ]
                -
                paired_df[
                    old_stress
                ]
            )

            # ------------------------------------------
            # Probabilities are stored as fractions.
            #
            # Convert their paired difference to
            # percentage points for readability.
            # ------------------------------------------

            if (
                "probability"
                in column_name
            ):

                paired_df[
                    difference_column
                ] = (
                    paired_df[
                        difference_column
                    ]
                    * 100
                )

            mean_change, lower, upper = (
                bootstrap_metric(
                    paired_df[
                        [
                            "network_seed",
                            difference_column
                        ]
                    ],
                    difference_column
                )
            )

            paired_change_rows.append(
                {
                    "comparison":
                        (
                            f"{new_stress}"
                            f"_minus_"
                            f"{old_stress}"
                        ),

                    "metric":
                        metric_name,

                    "paired_scenarios":
                        len(
                            paired_df
                        ),

                    "networks":
                        paired_df[
                            "network_seed"
                        ]
                        .nunique(),

                    "mean_change":
                        mean_change,

                    "ci_lower":
                        lower,

                    "ci_upper":
                        upper
                }
            )

    paired_change_df = (
        pd.DataFrame(
            paired_change_rows
        )
    )

    # ==================================================
    # 3. Regime × stress breakdown
    # ==================================================

    regime_stress_summary = (
        df
        .groupby(
            [
                "stress_level",
                "scenario"
            ]
        )
        .agg(
            networks=(
                "network_seed",
                "nunique"
            ),

            mean_improvement=(
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

            mean_loss_probability=(
                "loss_probability",
                "mean"
            ),

            mean_ml_std=(
                "ml_realized_std",
                "mean"
            )
        )
        .reset_index()
    )

    # ==================================================
    # 4. Stress sensitivity of individual
    # decision scenarios
    #
    # Which scenarios deteriorate most from
    # mild -> severe?
    # ==================================================

    sensitivity_columns = [
        "mean_ml_improvement_vs_free_flow_percent",
        "loss_probability",
        "ml_realized_std"
    ]

    sensitivity_frames = []

    for column_name in (
        sensitivity_columns
    ):

        pivot_df = (
            df.pivot(
                index=[
                    "network_seed",
                    "scenario"
                ],

                columns=(
                    "stress_level"
                ),

                values=(
                    column_name
                )
            )
            .reset_index()
        )

        pivot_df[
            "severe_minus_mild"
        ] = (
            pivot_df[
                "severe"
            ]
            -
            pivot_df[
                "mild"
            ]
        )

        pivot_df[
            "metric"
        ] = column_name

        sensitivity_frames.append(
            pivot_df
        )

    sensitivity_df = pd.concat(
        sensitivity_frames,
        ignore_index=True
    )

    # ==================================================
    # 5. Simple monotonic diagnostics
    #
    # We are NOT doing hypothesis testing here.
    # This just describes how often scenario-level
    # reliability worsens as stress increases.
    # ==================================================

    loss_pivot = (
        df.pivot(
            index=[
                "network_seed",
                "scenario"
            ],

            columns="stress_level",

            values="loss_probability"
        )
    )

    loss_monotonic_increase = (
        (
            loss_pivot[
                "moderate"
            ]
            >=
            loss_pivot[
                "mild"
            ]
        )
        &
        (
            loss_pivot[
                "severe"
            ]
            >=
            loss_pivot[
                "moderate"
            ]
        )
    )

    std_pivot = (
        df.pivot(
            index=[
                "network_seed",
                "scenario"
            ],

            columns="stress_level",

            values="ml_realized_std"
        )
    )

    std_monotonic_increase = (
        (
            std_pivot[
                "moderate"
            ]
            >=
            std_pivot[
                "mild"
            ]
        )
        &
        (
            std_pivot[
                "severe"
            ]
            >=
            std_pivot[
                "moderate"
            ]
        )
    )

    # ==================================================
    # Save outputs
    # ==================================================

    stress_bootstrap_df.to_csv(
        "data/robustness_stress_cluster_bootstrap.csv",
        index=False
    )

    paired_change_df.to_csv(
        "data/robustness_stress_paired_changes.csv",
        index=False
    )

    regime_stress_summary.to_csv(
        "data/robustness_regime_stress_summary.csv",
        index=False
    )

    sensitivity_df.to_csv(
        "data/robustness_scenario_sensitivity.csv",
        index=False
    )

    # ==================================================
    # Report
    # ==================================================

    print(
        "\n"
        "========================================"
    )

    print(
        "ROBUSTNESS STRESS ANALYSIS"
    )

    print(
        "========================================"
    )

    print(
        f"Independent networks: "
        f"{df['network_seed'].nunique()}"
    )

    print(
        f"Decision scenarios: "
        f"{df[['network_seed', 'scenario']].drop_duplicates().shape[0]}"
    )

    # ==================================================
    # Stress bootstrap
    # ==================================================

    print(
        "\n"
        "========================================"
    )

    print(
        "STRESS-LEVEL NETWORK-CLUSTER BOOTSTRAP"
    )

    print(
        "========================================"
    )

    print(
        stress_bootstrap_df.to_string(
            index=False
        )
    )

    # ==================================================
    # Paired changes
    # ==================================================

    print(
        "\n"
        "========================================"
    )

    print(
        "PAIRED STRESS CHANGES"
    )

    print(
        "========================================"
    )

    print(
        paired_change_df.to_string(
            index=False
        )
    )

    # ==================================================
    # Regime × stress
    # ==================================================

    print(
        "\n"
        "========================================"
    )

    print(
        "TRAFFIC REGIME × STRESS"
    )

    print(
        "========================================"
    )

    print(
        regime_stress_summary.to_string(
            index=False
        )
    )

    # ==================================================
    # Monotonic diagnostics
    # ==================================================

    print(
        "\n"
        "========================================"
    )

    print(
        "MONOTONIC STRESS DIAGNOSTICS"
    )

    print(
        "========================================"
    )

    print(
        "Loss probability non-decreasing "
        "from mild → moderate → severe:"
    )

    print(
        f"{loss_monotonic_increase.sum()}/"
        f"{len(loss_monotonic_increase)} "
        f"scenarios"
    )

    print(
        "\nML realized standard deviation "
        "non-decreasing:"
    )

    print(
        f"{std_monotonic_increase.sum()}/"
        f"{len(std_monotonic_increase)} "
        f"scenarios"
    )

    print(
        "\nSaved:"
    )

    print(
        "data/robustness_stress_cluster_bootstrap.csv"
    )

    print(
        "data/robustness_stress_paired_changes.csv"
    )

    print(
        "data/robustness_regime_stress_summary.csv"
    )

    print(
        "data/robustness_scenario_sensitivity.csv"
    )


if __name__ == "__main__":
    main()

    