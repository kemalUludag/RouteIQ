import time
import pandas as pd
import matplotlib.pyplot as plt

from src.data_generator import generate_cvrp_instance
from src.distance import create_distance_matrix
from src.routing import calculate_route_distance
from src.ortools_solver import solve_cvrp_ortools
from src.validation import validate_solution


num_customers = 50
num_vehicles = 10
vehicle_capacity = 15

time_limits = [1, 3, 5]

seeds = range(3)

results = []


for seed in seeds:

    # -------------------------
    # Generate the same instance
    # for each time limit
    # -------------------------

    depot, customers, demands = generate_cvrp_instance(
        num_customers,
        num_vehicles,
        vehicle_capacity,
        seed=seed
    )

    points = [depot] + customers

    distance_matrix = create_distance_matrix(points)


    for time_limit in time_limits:

        # -------------------------
        # Solve with OR-Tools
        # -------------------------

        start_time = time.perf_counter()

        routes = solve_cvrp_ortools(
            distance_matrix,
            demands,
            num_vehicles,
            vehicle_capacity,
            time_limit_seconds=time_limit
        )

        end_time = time.perf_counter()

        runtime = end_time - start_time


        # -------------------------
        # Evaluate solution
        # -------------------------

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


        # -------------------------
        # Store result
        # -------------------------

        results.append({
            "seed": seed,
            "time_limit": time_limit,
            "distance": total_distance,
            "runtime": runtime,
            "valid": is_valid
        })


        print(
            f"Seed={seed}, "
            f"Limit={time_limit}s, "
            f"Distance={total_distance:.2f}, "
            f"Runtime={runtime:.3f}s, "
            f"Valid={is_valid}"
        )


# -------------------------
# DataFrame
# -------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "data/time_limit_results.csv",
    index=False
)


# -------------------------
# Summary
# -------------------------

summary_df = (
    results_df
    .groupby("time_limit")
    .agg(
        instances=("seed", "count"),
        mean_distance=("distance", "mean"),
        std_distance=("distance", "std"),
        mean_runtime=("runtime", "mean"),
        valid_rate=("valid", "mean")
    )
    .reset_index()
)

summary_df.to_csv(
    "data/time_limit_summary.csv",
    index=False
)


print()
print("=== TIME LIMIT SUMMARY ===")
print(summary_df)


# -------------------------
# Plot
# -------------------------

plt.figure()

plt.errorbar(
    summary_df["time_limit"],
    summary_df["mean_distance"],
    yerr=summary_df["std_distance"],
    marker="o",
    capsize=5
)

plt.xlabel("Solver Time Limit (seconds)")
plt.ylabel("Mean Route Distance")
plt.title("Solution Quality vs OR-Tools Search Time")

plt.savefig(
    "data/time_limit_quality.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()




