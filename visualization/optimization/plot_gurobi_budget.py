import pandas as pd
import matplotlib.pyplot as plt


def main():
    summary_df = pd.read_csv(
        "data/gurobi_budget_summary.csv"
    )

    # ==========================================
    # MIP gap vs computation budget
    # ==========================================

    plt.figure(
        figsize=(8, 5)
    )

    for num_customers in sorted(
        summary_df["num_customers"].unique()
    ):
        subset = summary_df[
            summary_df["num_customers"]
            == num_customers
        ]

        plt.plot(
            subset["time_limit_seconds"],
            subset["mean_mip_gap_percent"],
            marker="o",
            label=f"{num_customers} customers"
        )

    plt.xscale("log")

    plt.xlabel(
        "Gurobi Time Limit (seconds, log scale)"
    )

    plt.ylabel(
        "Mean MIP Gap (%)"
    )

    plt.title(
        "Optimality Gap vs Computation Budget"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "data/gurobi_budget_mip_gap.png",
        dpi=300
    )

    plt.close()

    # ==========================================
    # Optimality rate vs computation budget
    # ==========================================

    plt.figure(
        figsize=(8, 5)
    )

    for num_customers in sorted(
        summary_df["num_customers"].unique()
    ):
        subset = summary_df[
            summary_df["num_customers"]
            == num_customers
        ]

        plt.plot(
            subset["time_limit_seconds"],
            subset["optimal_rate"] * 100,
            marker="o",
            label=f"{num_customers} customers"
        )

    plt.xscale("log")

    plt.xlabel(
        "Gurobi Time Limit (seconds, log scale)"
    )

    plt.ylabel(
        "Optimal Solutions Proven (%)"
    )

    plt.title(
        "Optimality Rate vs Computation Budget"
    )

    plt.ylim(
        -5,
        105
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "data/gurobi_budget_optimality_rate.png",
        dpi=300
    )

    plt.close()

    # ==========================================
    # Feasibility rate
    # ==========================================

    plt.figure(
        figsize=(8, 5)
    )

    for num_customers in sorted(
        summary_df["num_customers"].unique()
    ):
        subset = summary_df[
            summary_df["num_customers"]
            == num_customers
        ]

        plt.plot(
            subset["time_limit_seconds"],
            subset["feasible_rate"] * 100,
            marker="o",
            label=f"{num_customers} customers"
        )

    plt.xscale("log")

    plt.xlabel(
        "Gurobi Time Limit (seconds, log scale)"
    )

    plt.ylabel(
        "Feasible Solution Rate (%)"
    )

    plt.title(
        "Feasible Solution Rate vs Computation Budget"
    )

    plt.ylim(
        -5,
        105
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "data/gurobi_budget_feasibility_rate.png",
        dpi=300
    )

    plt.close()

    print("Saved:")
    print(
        "data/gurobi_budget_mip_gap.png"
    )
    print(
        "data/gurobi_budget_optimality_rate.png"
    )
    print(
        "data/gurobi_budget_feasibility_rate.png"
    )


if __name__ == "__main__":
    main()
    