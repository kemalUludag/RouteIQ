from src.problem_data import depot, customers
from src.distance import calculate_distance, create_distance_matrix
from src.routing import calculate_route_distance, find_best_route_bruteforce

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
