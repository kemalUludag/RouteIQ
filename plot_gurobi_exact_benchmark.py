import pandas as pd
import matplotlib.pyplot as plt


def main():
    summary_df = pd.read_csv(
        "data/gurobi_exact_summary.csv"
    )

    # ==========================================
    # Runtime scaling
    # ==========================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        summary_df["num_customers"],
        summary_df[
            "bruteforce_runtime_mean"
        ],
        marker="o",
        label="Brute Force"
    )

    plt.plot(
        summary_df["num_customers"],
        summary_df[
            "gurobi_runtime_mean"
        ],
        marker="o",
        label="Gurobi MILP"
    )

    plt.yscale(
        "log"
    )

    plt.xlabel(
        "Number of Customers"
    )

    plt.ylabel(
        "Mean Runtime (seconds, log scale)"
    )

    plt.title(
        "Exact CVRP Runtime Scaling"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "data/gurobi_exact_runtime_scaling.png",
        dpi=300
    )

    plt.close()

    # ==========================================
    # Gurobi branch-and-bound node count
    # ==========================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        summary_df["num_customers"],
        summary_df[
            "gurobi_node_count_mean"
        ],
        marker="o"
    )

    plt.xlabel(
        "Number of Customers"
    )

    plt.ylabel(
        "Mean Branch-and-Bound Nodes"
    )

    plt.title(
        "Gurobi Search Effort by Problem Size"
    )

    plt.tight_layout()

    plt.savefig(
        "data/gurobi_node_scaling.png",
        dpi=300
    )

    plt.close()

    print(
        "Saved:"
    )

    print(
        "data/gurobi_exact_runtime_scaling.png"
    )

    print(
        "data/gurobi_node_scaling.png"
    )


if __name__ == "__main__":
    main()

    