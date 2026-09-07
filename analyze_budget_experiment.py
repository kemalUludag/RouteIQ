import pandas as pd

from src.result_analysis import (
    compare_solver_to_baseline,
    bootstrap_mean_confidence_interval
)


results_df = pd.read_csv(
    "data/budget_experiment_results.csv"
)


# ==================================================
# Clean solver-level summary
# ==================================================

summary_data = (
    results_df.copy()
)

summary_data.loc[
    summary_data["feasible"] == False,
    "distance"
] = float("nan")


solver_summary = (
    summary_data
    .groupby([
        "num_customers",
        "solver"
    ])
    .agg(
        instances=(
            "instance",
            "count"
        ),
        mean_distance=(
            "distance",
            "mean"
        ),
        std_distance=(
            "distance",
            "std"
        ),
        mean_runtime=(
            "runtime_seconds",
            "mean"
        ),
        feasibility_rate=(
            "feasible",
            "mean"
        )
    )
    .reset_index()
)


solver_summary.to_csv(
    "data/budget_solver_summary.csv",
    index=False
)


# ==================================================
# Paired comparisons against Clarke-Wright
# ==================================================

challengers = [
    "OR-Tools-1s",
    "OR-Tools-3s",
    "OR-Tools-5s"
]

pairwise_frames = []


for challenger in challengers:
    comparison_df = (
        compare_solver_to_baseline(
            results_df=results_df,
            baseline_solver="Clarke-Wright",
            challenger_solver=challenger
        )
    )

    pairwise_frames.append(
        comparison_df
    )


pairwise_results = pd.concat(
    pairwise_frames,
    ignore_index=True
)


pairwise_results.to_csv(
    "data/budget_pairwise_results.csv",
    index=False
)


pairwise_summary = (
    pairwise_results
    .groupby([
        "num_customers",
        "challenger_solver"
    ])
    .agg(
        paired_instances=(
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

confidence_interval_rows = []


for (
    num_customers,
    challenger_solver
), group in pairwise_results.groupby(
    [
        "num_customers",
        "challenger_solver"
    ]
):
    lower, upper = (
        bootstrap_mean_confidence_interval(
            group[
                "objective_improvement_percent"
            ],
            confidence_level=0.95,
            num_bootstrap_samples=10000,
            seed=42
        )
    )

    confidence_interval_rows.append({
        "num_customers":
            num_customers,
        "challenger_solver":
            challenger_solver,
        "ci_95_lower":
            lower,
        "ci_95_upper":
            upper
    })


confidence_intervals = pd.DataFrame(
    confidence_interval_rows
)


pairwise_summary = (
    pairwise_summary.merge(
        confidence_intervals,
        on=[
            "num_customers",
            "challenger_solver"
        ],
        how="left"
    )
)

outcome_counts = (
    pairwise_results
    .groupby([
        "num_customers",
        "challenger_solver",
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


pairwise_summary = (
    pairwise_summary.merge(
        outcome_counts[
            [
                "num_customers",
                "challenger_solver",
                "win",
                "tie",
                "loss"
            ]
        ],
        on=[
            "num_customers",
            "challenger_solver"
        ],
        how="left"
    )
)


pairwise_summary["win_rate"] = (
    pairwise_summary["win"]
    / pairwise_summary["paired_instances"]
)

pairwise_summary["tie_rate"] = (
    pairwise_summary["tie"]
    / pairwise_summary["paired_instances"]
)

pairwise_summary["loss_rate"] = (
    pairwise_summary["loss"]
    / pairwise_summary["paired_instances"]
)


pairwise_summary.to_csv(
    "data/budget_pairwise_summary.csv",
    index=False
)


print(
    "=== SOLVER SUMMARY ==="
)

print(
    solver_summary.to_string(
        index=False
    )
)

print()

print(
    "=== OR-TOOLS BUDGET VS CLARKE-WRIGHT ==="
)

print(
    pairwise_summary.to_string(
        index=False
    )
)

# ==================================================
# Direct OR-Tools budget comparisons
# ==================================================

budget_comparisons = [
    (
        "OR-Tools-1s",
        "OR-Tools-3s",
        "3s_vs_1s"
    ),
    (
        "OR-Tools-3s",
        "OR-Tools-5s",
        "5s_vs_3s"
    ),
    (
        "OR-Tools-1s",
        "OR-Tools-5s",
        "5s_vs_1s"
    )
]


budget_comparison_rows = []


for (
    baseline_solver,
    challenger_solver,
    comparison_name
) in budget_comparisons:

    comparison_df = (
        compare_solver_to_baseline(
            results_df=results_df,
            baseline_solver=baseline_solver,
            challenger_solver=challenger_solver
        )
    )

    for (
        num_customers,
        group
    ) in comparison_df.groupby(
        "num_customers"
    ):
        lower, upper = (
            bootstrap_mean_confidence_interval(
                group[
                    "objective_improvement_percent"
                ],
                confidence_level=0.95,
                num_bootstrap_samples=10000,
                seed=42
            )
        )

        budget_comparison_rows.append({
            "num_customers":
                num_customers,
            "comparison":
                comparison_name,
            "instances":
                len(group),
            "mean_improvement_percent":
                group[
                    "objective_improvement_percent"
                ].mean(),
            "median_improvement_percent":
                group[
                    "objective_improvement_percent"
                ].median(),
            "min_improvement_percent":
                group[
                    "objective_improvement_percent"
                ].min(),
            "max_improvement_percent":
                group[
                    "objective_improvement_percent"
                ].max(),
            "ci_95_lower":
                lower,
            "ci_95_upper":
                upper,
            "wins":
                (
                    group["outcome"]
                    == "win"
                ).sum(),
            "ties":
                (
                    group["outcome"]
                    == "tie"
                ).sum(),
            "losses":
                (
                    group["outcome"]
                    == "loss"
                ).sum()
        })


budget_comparison_summary = (
    pd.DataFrame(
        budget_comparison_rows
    )
)


budget_comparison_summary.to_csv(
    "data/budget_direct_comparison.csv",
    index=False
)


print()

print(
    "=== DIRECT OR-TOOLS BUDGET COMPARISON ==="
)

print(
    budget_comparison_summary.to_string(
        index=False
    )
)

