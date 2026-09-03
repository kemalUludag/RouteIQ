from src.problem_data import (
    depot,
    customers,
    demands,
    num_vehicles,
    vehicle_capacity
)

from src.distance import create_distance_matrix

from src.routing import find_best_cvrp_bruteforce

from src.validation import validate_solution

from src.reporting import print_solution_report


points = [depot] + customers

distance_matrix = create_distance_matrix(points)

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

print("=== RouteIQ ===")
print(f"Solution valid: {is_valid}")
print(f"Validation: {validation_message}")
print()

print_solution_report(
    best_routes,
    distance_matrix,
    demands,
    vehicle_capacity
)
