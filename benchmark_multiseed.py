import time
import pandas as pd
import matplotlib.pyplot as plt


from src.data_generator import generate_cvrp_instance
from src.distance import create_distance_matrix
from src.routing import calculate_route_distance
from src.ortools_solver import solve_cvrp_ortools
from src.validation import validate_solution


customer_sizes = [10, 20, 50]

seeds = range(5)

vehicle_capacity = 15
solver_time_limit = 1

results = []


for num_customers in customer_sizes:

    num_vehicles = max(
        2,
        (num_customers + 4) // 5
    )

    for seed in seeds:

        depot, customers, demands = generate_cvrp_instance(
            num_customers,
            num_vehicles,
            vehicle_capacity,
            seed=seed
        )

        points = [depot] + customers

        distance_matrix = create_distance_matrix(points)


        start_time = time.perf_counter()

        routes = solve_cvrp_ortools(
            distance_matrix,
            demands,
            num_vehicles,
            vehicle_capacity,
            time_limit_seconds=solver_time_limit
        )

        end_time = time.perf_counter()

        runtime = end_time - start_time


        if routes is None:
            total_distance = None
            is_valid = False

        else:
            total_distance = sum(
                calculate_route_distance(
                    route,
                    distance_matrix
                )
                for route in routes
            )

            is_valid, _ = validate_solution(
                routes,
                num_customers,
                demands,
                vehicle_capacity,
                num_vehicles
            )


        results.append({
            "num_customers": num_customers,
            "num_vehicles": num_vehicles,
            "seed": seed,
            "distance": total_distance,
            "runtime": runtime,
            "valid": is_valid
        })

        print(
            f"Customers={num_customers}, "
            f"Seed={seed}, "
            f"Distance={total_distance:.2f}, "
            f"Runtime={runtime:.3f}s, "
            f"Valid={is_valid}"
        )


results_df = pd.DataFrame(results)

results_df.to_csv(
    "data/multiseed_results.csv",
    index=False
)


summary_df = (
    results_df
    .groupby("num_customers")
    .agg(
        instances=("seed", "count"),
        mean_distance=("distance", "mean"),
        std_distance=("distance", "std"),
        min_distance=("distance", "min"),
        max_distance=("distance", "max"),
        mean_runtime=("runtime", "mean"),
        valid_rate=("valid", "mean")
    )
    .reset_index()
)


summary_df.to_csv(
    "data/multiseed_summary.csv",
    index=False
)


print()
print("=== MULTI-SEED SUMMARY ===")
print(summary_df)

plt.figure()

plt.errorbar(
    summary_df["num_customers"],
    summary_df["mean_distance"],
    yerr=summary_df["std_distance"],
    marker="o",
    capsize=5
)

plt.xlabel("Number of Customers")
plt.ylabel("Mean Route Distance")
plt.title("OR-Tools CVRP Performance Across Random Instances")

plt.savefig(
    "data/multiseed_distance_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


