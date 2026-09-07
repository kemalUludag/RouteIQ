import pandas as pd
import matplotlib.pyplot as plt


pairwise_df = pd.read_csv(
    "data/budget_pairwise_summary.csv"
)


budget_map = {
    "OR-Tools-1s": 1,
    "OR-Tools-3s": 3,
    "OR-Tools-5s": 5
}


pairwise_df[
    "time_limit_seconds"
] = pairwise_df[
    "challenger_solver"
].map(
    budget_map
)


# ==================================================
# 1. Mean objective improvement vs computation budget
# ==================================================

for num_customers in sorted(
    pairwise_df["num_customers"].unique()
):
    subset = pairwise_df[
        pairwise_df["num_customers"]
        == num_customers
    ]

    plt.plot(
        subset["time_limit_seconds"],
        subset["mean_improvement_percent"],
        marker="o",
        label=f"{num_customers} customers"
    )


plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel(
    "OR-Tools Time Limit (seconds)"
)

plt.ylabel(
    "Mean Improvement over Clarke-Wright (%)"
)

plt.title(
    "Solution Quality vs Computation Budget"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "data/budget_quality_vs_time.png",
    dpi=300
)

plt.close()


# ==================================================
# 2. Loss rate vs computation budget
# ==================================================

for num_customers in sorted(
    pairwise_df["num_customers"].unique()
):
    subset = pairwise_df[
        pairwise_df["num_customers"]
        == num_customers
    ]

    plt.plot(
        subset["time_limit_seconds"],
        subset["loss_rate"] * 100,
        marker="o",
        label=f"{num_customers} customers"
    )


plt.xlabel(
    "OR-Tools Time Limit (seconds)"
)

plt.ylabel(
    "Loss Rate vs Clarke-Wright (%)"
)

plt.title(
    "Solver Reliability vs Computation Budget"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "data/budget_loss_rate.png",
    dpi=300
)

plt.close()


print("Saved:")
print("data/budget_quality_vs_time.png")
print("data/budget_loss_rate.png")


