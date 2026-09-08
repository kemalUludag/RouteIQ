import pandas as pd

from src.result_analysis import (
    bootstrap_mean_confidence_interval
)


COMPARISONS = [
    (0.05, 0.10),
    (0.10, 0.50),
    (0.50, 1.00),
    (0.05, 1.00)
]


def main():
    results_df = pd.read_csv(
        "data/gurobi_budget_results.csv"
    )

    analysis_rows = []

    for num_customers in sorted(
        results_df["num_customers"].unique()
    ):
        size_df = results_df[
            results_df["num_customers"]
            == num_customers
        ]

        for old_budget, new_budget in COMPARISONS:

            old_df = (
                size_df[
                    size_df["time_limit_seconds"]
                    == old_budget
                ]
                .copy()
            )

            new_df = (
                size_df[
                    size_df["time_limit_seconds"]
                    == new_budget
                ]
                .copy()
            )

            paired = pd.merge(
                old_df,
                new_df,
                on=[
                    "instance",
                    "seed",
                    "num_customers"
                ],
                suffixes=(
                    "_old",
                    "_new"
                )
            )

            # ------------------------------------------
            # Only compare incumbent quality when both
            # runs produced feasible solutions
            # ------------------------------------------

            feasible_pairs = paired[
                (
                    paired["feasible_old"] == True
                )
                &
                (
                    paired["feasible_new"] == True
                )
            ].copy()

            if feasible_pairs.empty:
                continue

            # Positive = new/larger budget is better
            feasible_pairs[
                "incumbent_improvement_percent"
            ] = (
                (
                    feasible_pairs["distance_old"]
                    - feasible_pairs["distance_new"]
                )
                /
                feasible_pairs["distance_old"]
                * 100
            )

            # Positive = lower bound became stronger
            feasible_pairs[
                "bound_improvement_percent"
            ] = (
                (
                    feasible_pairs["best_bound_new"]
                    - feasible_pairs["best_bound_old"]
                )
                /
                feasible_pairs[
                    "best_bound_old"
                ].abs()
                * 100
            )

            # Positive = MIP gap decreased
            feasible_pairs[
                "gap_reduction_percentage_points"
            ] = (
                feasible_pairs[
                    "mip_gap_percent_old"
                ]
                -
                feasible_pairs[
                    "mip_gap_percent_new"
                ]
            )

            # ------------------------------------------
            # Bootstrap confidence intervals
            # ------------------------------------------

            incumbent_values = (
                feasible_pairs[
                    "incumbent_improvement_percent"
                ]
                .dropna()
                .to_numpy()
            )

            bound_values = (
                feasible_pairs[
                    "bound_improvement_percent"
                ]
                .dropna()
                .to_numpy()
            )

            gap_values = (
                feasible_pairs[
                    "gap_reduction_percentage_points"
                ]
                .dropna()
                .to_numpy()
            )

            incumbent_ci = (
                bootstrap_mean_confidence_interval(
                    incumbent_values
                )
            )

            bound_ci = (
                bootstrap_mean_confidence_interval(
                    bound_values
                )
            )

            gap_ci = (
                bootstrap_mean_confidence_interval(
                    gap_values
                )
            )

            # ------------------------------------------
            # Win / tie / loss for incumbent
            # ------------------------------------------

            tolerance = 1e-9

            wins = (
                feasible_pairs[
                    "incumbent_improvement_percent"
                ]
                > tolerance
            ).sum()

            ties = (
                feasible_pairs[
                    "incumbent_improvement_percent"
                ]
                .abs()
                <= tolerance
            ).sum()

            losses = (
                feasible_pairs[
                    "incumbent_improvement_percent"
                ]
                < -tolerance
            ).sum()

            analysis_rows.append(
                {
                    "num_customers":
                        num_customers,

                    "old_budget":
                        old_budget,

                    "new_budget":
                        new_budget,

                    "paired_instances":
                        len(feasible_pairs),

                    "mean_incumbent_improvement_percent":
                        incumbent_values.mean(),

                    "incumbent_ci_lower":
                        incumbent_ci[0],

                    "incumbent_ci_upper":
                        incumbent_ci[1],

                    "mean_bound_improvement_percent":
                        bound_values.mean(),

                    "bound_ci_lower":
                        bound_ci[0],

                    "bound_ci_upper":
                        bound_ci[1],

                    "mean_gap_reduction_pp":
                        gap_values.mean(),

                    "gap_ci_lower":
                        gap_ci[0],

                    "gap_ci_upper":
                        gap_ci[1],

                    "wins":
                        int(wins),

                    "ties":
                        int(ties),

                    "losses":
                        int(losses)
                }
            )

    analysis_df = pd.DataFrame(
        analysis_rows
    )

    analysis_df.to_csv(
        "data/gurobi_budget_paired_analysis.csv",
        index=False
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "GUROBI PAIRED BUDGET ANALYSIS"
    )

    print(
        "========================================"
    )

    print(
        analysis_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()

    