import time
import pandas as pd
import matplotlib.pyplot as plt

from src.data_generator import generate_cvrp_instance
from src.distance import create_distance_matrix
from src.routing import (
    find_best_cvrp_bruteforce,
    calculate_route_distance
)
from src.ortools_solver import solve_cvrp_ortools


results = []

customer_sizes = [5, 6, 7, 8, 9, 10, 20, 50]

brute_force_max_customers = 9


for num_customers in customer_sizes:

    # -------------------------
    # Problem configuration
    # -------------------------

    num_vehicles = max(
        2,
        (num_customers + 4) // 5
    )

    vehicle_capacity = 15


    # -------------------------
    # Generate problem instance
    # -------------------------

    depot, customers, demands = generate_cvrp_instance(
        num_customers,
        num_vehicles,
        vehicle_capacity,
        seed=42
    )

    points = [depot] + customers

    distance_matrix = create_distance_matrix(points)


    # -------------------------
    # Brute-force benchmark
    # -------------------------

    if num_customers <= brute_force_max_customers:

        brute_force_start = time.perf_counter()

        brute_force_routes, brute_force_distance = (
            find_best_cvrp_bruteforce(
                num_customers,
                distance_matrix,
                demands,
                vehicle_capacity
            )
        )

        brute_force_end = time.perf_counter()

        brute_force_time = (
            brute_force_end - brute_force_start
        )

    else:
        brute_force_routes = None
        brute_force_distance = None
        brute_force_time = None


    # -------------------------
    # OR-Tools benchmark
    # -------------------------

    ortools_start = time.perf_counter()

    ortools_routes = solve_cvrp_ortools(
        distance_matrix,
        demands,
        num_vehicles,
        vehicle_capacity
    )

    ortools_end = time.perf_counter()

    ortools_time = (
        ortools_end - ortools_start
    )


    # -------------------------
    # OR-Tools route distance
    # -------------------------

    if ortools_routes is None:
        ortools_distance = None

    else:
        ortools_distance = sum(
            calculate_route_distance(
                route,
                distance_matrix
            )
            for route in ortools_routes
        )


    # -------------------------
    # Terminal output
    # -------------------------

    print(f"=== {num_customers} CUSTOMERS ===")
    print(f"Vehicles: {num_vehicles}")
    print(f"Vehicle capacity: {vehicle_capacity}")

    if brute_force_distance is None:
        print("Brute-force: Skipped")

    else:
        print(
            f"Brute-force: "
            f"{brute_force_distance:.2f} distance, "
            f"{brute_force_time:.6f} seconds"
        )

    if ortools_distance is None:
        print("OR-Tools: No feasible solution")

    else:
        print(
            f"OR-Tools:    "
            f"{ortools_distance:.2f} distance, "
            f"{ortools_time:.6f} seconds"
        )

    print()


    # -------------------------
    # Store experiment result
    # -------------------------

    results.append({
        "num_customers": num_customers,
        "num_vehicles": num_vehicles,
        "vehicle_capacity": vehicle_capacity,
        "brute_force_distance": brute_force_distance,
        "brute_force_time": brute_force_time,
        "ortools_distance": ortools_distance,
        "ortools_time": ortools_time
    })


# -------------------------
# Convert results to DataFrame
# -------------------------

results_df = pd.DataFrame(results)

print("=== BENCHMARK TABLE ===")
print(results_df)


# -------------------------
# Save CSV
# -------------------------

results_df.to_csv(
    "data/scaling_results.csv",
    index=False
)


# -------------------------
# Linear-scale runtime plot
# -------------------------

plt.figure()

plt.plot(
    results_df["num_customers"],
    results_df["brute_force_time"],
    marker="o",
    label="Brute Force"
)

plt.plot(
    results_df["num_customers"],
    results_df["ortools_time"],
    marker="o",
    label="OR-Tools"
)

plt.xlabel("Number of Customers")
plt.ylabel("Runtime (seconds)")
plt.title("CVRP Solver Runtime Comparison")

plt.legend()

plt.savefig(
    "data/scaling_runtime.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# -------------------------
# Log-scale runtime plot
# -------------------------

plt.figure()

plt.plot(
    results_df["num_customers"],
    results_df["brute_force_time"],
    marker="o",
    label="Brute Force"
)

plt.plot(
    results_df["num_customers"],
    results_df["ortools_time"],
    marker="o",
    label="OR-Tools"
)

plt.yscale("log")

plt.xlabel("Number of Customers")
plt.ylabel("Runtime (seconds, log scale)")
plt.title("CVRP Solver Runtime Comparison - Log Scale")

plt.legend()

plt.savefig(
    "data/scaling_runtime_log.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

