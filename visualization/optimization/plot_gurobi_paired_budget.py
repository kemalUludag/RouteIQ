import pandas as pd
import matplotlib.pyplot as plt


def main():
    analysis_df = pd.read_csv(
        "data/gurobi_budget_paired_analysis.csv"
    )

    # We focus on the broadest budget comparison:
    # 0.05 seconds -> 1.00 second

    comparison_df = analysis_df[
        (
            analysis_df["old_budget"] == 0.05
        )
        &
        (
            analysis_df["new_budget"] == 1.00
        )
    ].copy()

    # ==========================================
    # Incumbent improvement
    # ==========================================

    lower_error = (
        comparison_df[
            "mean_incumbent_improvement_percent"
        ]
        -
        comparison_df[
            "incumbent_ci_lower"
        ]
    )

    upper_error = (
        comparison_df[
            "incumbent_ci_upper"
        ]
        -
        comparison_df[
            "mean_incumbent_improvement_percent"
        ]
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.errorbar(
        comparison_df["num_customers"],
        comparison_df[
            "mean_incumbent_improvement_percent"
        ],
        yerr=[
            lower_error,
            upper_error
        ],
        marker="o",
        capsize=5
    )

    plt.xlabel(
        "Number of Customers"
    )

    plt.ylabel(
        "Mean Incumbent Improvement (%)"
    )

    plt.title(
        "Value of Additional MILP Search Time"
    )

    plt.tight_layout()

    plt.savefig(
        "data/gurobi_paired_incumbent_improvement.png",
        dpi=300
    )

    plt.close()

    # ==========================================
    # Gap reduction
    # ==========================================

    gap_lower_error = (
        comparison_df[
            "mean_gap_reduction_pp"
        ]
        -
        comparison_df[
            "gap_ci_lower"
        ]
    )

    gap_upper_error = (
        comparison_df[
            "gap_ci_upper"
        ]
        -
        comparison_df[
            "mean_gap_reduction_pp"
        ]
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.errorbar(
        comparison_df["num_customers"],
        comparison_df[
            "mean_gap_reduction_pp"
        ],
        yerr=[
            gap_lower_error,
            gap_upper_error
        ],
        marker="o",
        capsize=5
    )

    plt.xlabel(
        "Number of Customers"
    )

    plt.ylabel(
        "Mean MIP Gap Reduction (percentage points)"
    )

    plt.title(
        "Optimality Certificate Improvement with Search Time"
    )

    plt.tight_layout()

    plt.savefig(
        "data/gurobi_paired_gap_reduction.png",
        dpi=300
    )

    plt.close()

    print("Saved:")
    print(
        "data/gurobi_paired_incumbent_improvement.png"
    )
    print(
        "data/gurobi_paired_gap_reduction.png"
    )


if __name__ == "__main__":
    main()

    