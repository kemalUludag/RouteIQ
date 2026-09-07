import pandas as pd
import numpy as np

def compare_solver_to_baseline(
    results_df: pd.DataFrame,
    baseline_solver: str,
    challenger_solver: str
) -> pd.DataFrame:
    baseline_df = (
        results_df[
            (
                results_df["solver"]
                == baseline_solver
            )
            & (
                results_df["feasible"]
                == True
            )
        ]
        [
            [
                "instance",
                "num_customers",
                "distance",
                "runtime_seconds"
            ]
        ]
        .rename(
            columns={
                "distance":
                    "baseline_distance",
                "runtime_seconds":
                    "baseline_runtime"
            }
        )
    )

    challenger_df = (
        results_df[
            (
                results_df["solver"]
                == challenger_solver
            )
            & (
                results_df["feasible"]
                == True
            )
         ]
        [
            [
                "instance",
                "num_customers",
                "distance",
                "runtime_seconds"
            ]
        ]
        .rename(
            columns={
                "distance":
                    "challenger_distance",
                "runtime_seconds":
                    "challenger_runtime"
            }
        )
    )

    comparison_df = baseline_df.merge(
        challenger_df,
        on=[
            "instance",
            "num_customers"
        ],
        how="inner"
    )

    comparison_df[
        "objective_improvement_percent"
    ] = (
        (
            comparison_df["baseline_distance"]
            - comparison_df["challenger_distance"]
        )
        / comparison_df["baseline_distance"]
    ) * 100


    comparison_df.loc[
        comparison_df[
            "objective_improvement_percent"
        ].abs() < 1e-9,
        "objective_improvement_percent"
    ] = 0.0

    tolerance = 1e-9

    comparison_df["outcome"] = "tie"

    comparison_df.loc[
        comparison_df[
            "objective_improvement_percent"
        ] > tolerance,
        "outcome"
    ] = "win"

    comparison_df.loc[
        comparison_df[
            "objective_improvement_percent"
        ] < -tolerance,
        "outcome"
    ]= "loss"


    comparison_df[
        "runtime_ratio"
    ] = (
        comparison_df["challenger_runtime"]
        / comparison_df["baseline_runtime"]
    )
    
    comparison_df[
        "baseline_solver"
    ] = baseline_solver

    comparison_df[
        "challenger_solver"
    ] = challenger_solver

    return comparison_df


import numpy as np


def bootstrap_mean_confidence_interval(
    values,
    confidence_level=0.95,
    num_bootstrap_samples=10000,
    seed=42
):
    values = np.asarray(
        values,
        dtype=float
    )

    if len(values) == 0:
        return None, None

    rng = np.random.default_rng(
        seed
    )

    bootstrap_means = []

    for _ in range(
        num_bootstrap_samples
    ):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True
        )

        bootstrap_means.append(
            sample.mean()
        )

    alpha = (
        1 - confidence_level
    )

    lower = np.percentile(
        bootstrap_means,
        100 * alpha / 2
    )

    upper = np.percentile(
        bootstrap_means,
        100 * (
            1 - alpha / 2
        )
    )

    return lower, upper

