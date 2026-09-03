import time

from src.data_generator import generate_cvrp_instance
from src.distance import create_distance_matrix
from src.routing import (
    find_best_cvrp_bruteforce,
    calculate_route_distance
)
from src.ortools_solver import solve_cvrp_ortools


customer_sizes = [5, 6, 7, 8, 9]

num_vehicles = 2
vehicle_capacity = 20


for num_customers in customer_sizes:
    depot, customers, demands = generate_cvrp_instance(
        num_customers,
        num_vehicles,
        vehicle_capacity,
        seed=42
    )

    points = [depot] + customers

    distance_matrix = create_distance_matrix(points)


    # Brute-force benchmark

    brute_force_start = time.perf_counter()

    brute_force_routes, brute_force_distance = find_best_cvrp_bruteforce(
        num_customers,
        distance_matrix,
        demands,
        vehicle_capacity
    )

    brute_force_end = time.perf_counter()

    brute_force_time = (
        brute_force_end - brute_force_start
    )


    # OR-Tools benchmark

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


    print(f"=== {num_customers} CUSTOMERS ===")

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

