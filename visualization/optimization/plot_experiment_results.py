import pandas as pd
import matplotlib.pyplot as plt


results_df = pd.read_csv(
    "data/experiment_results.csv"
)


# ==================================================
# 1. Runtime scaling
# ==================================================

runtime_summary = (
    results_df
    .groupby([
        "num_customers",
        "solver"
    ])["runtime_seconds"]
    .mean()
    .reset_index()
)


for solver_name in runtime_summary["solver"].unique():
    solver_data = runtime_summary[
        runtime_summary["solver"]
        == solver_name
    ]

    plt.plot(
        solver_data["num_customers"],
        solver_data["runtime_seconds"],
        marker="o",
        label=solver_name
    )


plt.yscale("log")

plt.xlabel(
    "Number of Customers"
)

plt.ylabel(
    "Mean Runtime (seconds, log scale)"
)

plt.title(
    "Solver Runtime Scaling"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "data/experiment_runtime_scaling.png",
    dpi=300
)

plt.close()


# ==================================================
# 2. OR-Tools vs exact objective
# ==================================================

small_instances = results_df[
    results_df["num_customers"] <= 9
]

objective_table = (
    small_instances
    .pivot(
        index="instance",
        columns="solver",
        values="distance"
    )
    .dropna()
)


plt.scatter(
    objective_table["Brute Force"],
    objective_table["OR-Tools"]
)


minimum = min(
    objective_table["Brute Force"].min(),
    objective_table["OR-Tools"].min()
)

maximum = max(
    objective_table["Brute Force"].max(),
    objective_table["OR-Tools"].max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel(
    "Exact Objective — Brute Force"
)

plt.ylabel(
    "OR-Tools Objective"
)

plt.title(
    "OR-Tools Solution Quality vs Exact Optimum"
)

plt.tight_layout()

plt.savefig(
    "data/experiment_objective_parity.png",
    dpi=300
)

plt.close()


print(
    "Saved:"
)

print(
    "data/experiment_runtime_scaling.png"
)

print(
    "data/experiment_objective_parity.png"
)

