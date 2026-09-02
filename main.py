from src.problem_data import (
    depot,
    customers,
    demands,
    num_vehicles,
    vehicle_capacity
)
from src.distance import calculate_distance, create_distance_matrix
from src.routing import (
    calculate_route_distance,
    find_best_route_bruteforce,
    calculate_route_demand,
    is_route_feasible,
    find_best_cvrp_bruteforce
)
from src.validation import validate_solution
from src.reporting import print_solution_report


print(depot)
print(customers)


distance = calculate_distance(depot, customers[0])

print(distance)
points = [depot] + customers
distance_matrix = create_distance_matrix(points)

for row in distance_matrix:
    print(row)


route = [0, 1, 2, 3, 4, 5, 0]

route_distance = calculate_route_distance(route, distance_matrix)

print("Route:", route)
print("Total route distance:", route_distance)   

best_route, best_distance = find_best_route_bruteforce(
    len(customers),
    distance_matrix
)

print("Best route:", best_route)
print("Best distance:", best_distance)



feasible_route = [0, 1, 3, 0]
infeasible_route = [0, 2, 3, 4, 0]

print(
    "Feasible route demand:",
    calculate_route_demand(feasible_route, demands)
)

print(
    "Feasible route?",
    is_route_feasible(feasible_route, demands, vehicle_capacity)
)

print(
    "Infeasible route demand:",
    calculate_route_demand(infeasible_route, demands)
)

print(
    "Infeasible route?",
    is_route_feasible(infeasible_route, demands, vehicle_capacity)
)


best_routes, best_cvrp_distance = find_best_cvrp_bruteforce(
    len(customers),
    distance_matrix,
    demands,
    vehicle_capacity
)

print("Best CVRP routes:", best_routes)
print("Best CVRP total distance:", best_cvrp_distance)

is_valid, validation_message = validate_solution(
    best_routes,
    len(customers),
    demands,
    vehicle_capacity,
    num_vehicles
)

print("Solution valid?", is_valid)
print("Validation message:", validation_message)


print()
print("=== CVRP SOLUTION REPORT ===")

print_solution_report(
    best_routes,
    distance_matrix,
    demands,
    vehicle_capacity
)
