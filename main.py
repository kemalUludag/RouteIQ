from src.problem_data import (
    depot,
    customers,
    demands,
    num_vehicles,
    vehicle_capacity
)

from src.distance import create_distance_matrix
import time
from src.routing import (
    find_best_cvrp_bruteforce,
    calculate_route_distance
)

from src.validation import validate_solution
from src.reporting import print_solution_report
from src.ortools_solver import solve_cvrp_ortools


points = [depot] + customers

distance_matrix = create_distance_matrix(points)


# -------------------------
# Brute-force solution
# -------------------------
brute_force_start = time.perf_counter()
best_routes, best_total_distance = find_best_cvrp_bruteforce(
    len(customers),
    distance_matrix,
    demands,
    vehicle_capacity
)

is_valid, validation_message = validate_solution(
    best_routes,
    len(customers),
    demands,
    vehicle_capacity,
    num_vehicles
)

print("=== BRUTE-FORCE CVRP ===")
print(f"Solution valid: {is_valid}")
print(f"Validation: {validation_message}")
print(f"Best routes: {best_routes}")
print(f"Total distance: {best_total_distance:.2f}")
print()

print_solution_report(
    best_routes,
    distance_matrix,
    demands,
    vehicle_capacity
)
brute_force_end = time.perf_counter()

brute_force_time = brute_force_end - brute_force_start

# -------------------------
# OR-Tools solution
# -------------------------
ortools_start = time.perf_counter()
ortools_routes = solve_cvrp_ortools(
    distance_matrix,
    demands,
    num_vehicles,
    vehicle_capacity
)
ortools_end = time.perf_counter()

ortools_time = ortools_end - ortools_start

if ortools_routes is None:
    print("OR-Tools could not find a feasible solution.")

else:
    ortools_total_distance = sum(
        calculate_route_distance(route, distance_matrix)
        for route in ortools_routes
    )

    ortools_valid, ortools_message = validate_solution(
        ortools_routes,
        len(customers),
        demands,
        vehicle_capacity,
        num_vehicles
    )

    print("=== OR-TOOLS CVRP ===")
    print(f"Solution valid: {ortools_valid}")
    print(f"Validation: {ortools_message}")
    print(f"Best routes: {ortools_routes}")
    print(f"Total distance: {ortools_total_distance:.2f}")

print()
print("=== PERFORMANCE ===")
print(f"Brute-force time: {brute_force_time:.6f} seconds")
print(f"OR-Tools time: {ortools_time:.6f} seconds")
