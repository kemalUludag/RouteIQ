import pandas as pd

from src.result_analysis import (
    compare_solver_to_baseline
)


results_df = pd.read_csv(
    "data/experiment_results.csv"
)


comparison_df = compare_solver_to_baseline(
    results_df=results_df,
    baseline_solver="Clarke-Wright",
    challenger_solver="OR-Tools"
)


comparison_df.to_csv(
    "data/solver_pairwise_results.csv",
    index=False
)


summary_df = (
    comparison_df
    .groupby("num_customers")
    .agg(
        instances=(
            "instance",
            "count"
        ),
        mean_improvement_percent=(
            "objective_improvement_percent",
            "mean"
        ),
        median_improvement_percent=(
            "objective_improvement_percent",
            "median"
        ),
        std_improvement_percent=(
            "objective_improvement_percent",
            "std"
        ),
        min_improvement_percent=(
            "objective_improvement_percent",
            "min"
        ),
        max_improvement_percent=(
            "objective_improvement_percent",
            "max"
        ),
        mean_baseline_runtime=(
            "baseline_runtime",
            "mean"
        ),
        mean_challenger_runtime=(
            "challenger_runtime",
            "mean"
        )
    )
    .reset_index()
)

outcome_counts = (
    comparison_df
    .groupby([
        "num_customers",
        "outcome"
    ])
    .size()
    .unstack(
        fill_value=0
    )
    .reset_index()
)


for column in [
    "win",
    "tie",
    "loss"
]:
    if column not in outcome_counts.columns:
        outcome_counts[column] = 0


summary_df = summary_df.merge(
    outcome_counts[
        [
            "num_customers",
            "win",
            "tie",
            "loss"
        ]
    ],
    on="num_customers",
    how="left"
)


summary_df[
    "win_rate"
] = (
    summary_df["win"]
    / summary_df["instances"]
)

summary_df[
    "tie_rate"
] = (
    summary_df["tie"]
    / summary_df["instances"]
)

summary_df[
    "loss_rate"
] = (
    summary_df["loss"]
    / summary_df["instances"]
)

summary_df.to_csv(
    "data/solver_pairwise_summary.csv",
    index=False
)


print(
    "=== OR-TOOLS VS CLARKE-WRIGHT ==="
)

print(
    summary_df.to_string(
        index=False
    )
)

print()

print(
    "Saved:"
)

print(
    "data/solver_pairwise_results.csv"
)

print(
    "data/solver_pairwise_summary.csv"
)
