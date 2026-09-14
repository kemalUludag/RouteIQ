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


def cluster_bootstrap_mean_confidence_interval(
    df,
    cluster_column,
    value_column,
    confidence_level=0.95,
    num_bootstrap_samples=10000,
    seed=42
):
    # ==========================================
    # Validate arguments
    # ==========================================

    if cluster_column not in df.columns:
        raise ValueError(
            f"Unknown cluster column: "
            f"{cluster_column}"
        )

    if value_column not in df.columns:
        raise ValueError(
            f"Unknown value column: "
            f"{value_column}"
        )

    if not (
        0.0
        < confidence_level
        < 1.0
    ):
        raise ValueError(
            "confidence_level must be "
            "between 0 and 1."
        )

    if num_bootstrap_samples <= 0:
        raise ValueError(
            "num_bootstrap_samples must "
            "be positive."
        )

    # ==========================================
    # Keep only usable observations
    # ==========================================

    working_df = (
        df[
            [
                cluster_column,
                value_column
            ]
        ]
        .dropna()
        .copy()
    )

    if working_df.empty:
        raise ValueError(
            "No valid observations "
            "available for bootstrap."
        )

    # ==========================================
    # Identify independent clusters
    # ==========================================

    cluster_ids = (
        working_df[
            cluster_column
        ]
        .drop_duplicates()
        .to_numpy()
    )

    # Store each cluster's complete set
    # of repeated observations.
    #
    # In RouteIQ:
    #
    # network 0 -> 4 scenarios
    # network 1 -> 4 scenarios
    # ...
    #
    # These four observations stay together
    # whenever a network is resampled.
    # ==========================================

    cluster_values = {
        cluster_id:
            working_df.loc[
                working_df[
                    cluster_column
                ]
                == cluster_id,
                value_column
            ]
            .to_numpy(
                dtype=float
            )

        for cluster_id
        in cluster_ids
    }

    rng = np.random.default_rng(
        seed
    )

    bootstrap_means = np.empty(
        num_bootstrap_samples,
        dtype=float
    )

    # ==========================================
    # Cluster bootstrap
    # ==========================================

    for bootstrap_index in range(
        num_bootstrap_samples
    ):

        sampled_clusters = (
            rng.choice(
                cluster_ids,
                size=len(cluster_ids),
                replace=True
            )
        )

        # IMPORTANT:
        #
        # If network 7 is sampled twice,
        # all four observations belonging
        # to network 7 must appear twice.
        #
        # Using df.isin(...) would NOT do this
        # correctly because duplicates would
        # disappear.
        # ==========================================

        sampled_values = np.concatenate(
            [
                cluster_values[
                    cluster_id
                ]

                for cluster_id
                in sampled_clusters
            ]
        )

        bootstrap_means[
            bootstrap_index
        ] = np.mean(
            sampled_values
        )

    # ==========================================
    # Percentile confidence interval
    # ==========================================

    alpha = (
        1.0
        - confidence_level
    ) / 2.0

    lower_bound = np.quantile(
        bootstrap_means,
        alpha
    )

    upper_bound = np.quantile(
        bootstrap_means,
        1.0 - alpha
    )

    return (
        float(
            lower_bound
        ),
        float(
            upper_bound
        )
    )

