import itertools


def calculate_route_distance(route, distance_matrix):
    total_distance = 0

    for i in range(len(route) - 1):
        current_node = route[i]
        next_node = route[i + 1]

        total_distance += distance_matrix[current_node][next_node]

    return total_distance
def find_best_route_bruteforce(num_customers, distance_matrix):
    customer_nodes = list(range(1, num_customers + 1))

    best_route = None
    best_distance = float("inf")

    for permutation in itertools.permutations(customer_nodes):
        route = [0] + list(permutation) + [0]

        route_distance = calculate_route_distance(route, distance_matrix)

        if route_distance < best_distance:
            best_distance = route_distance
            best_route = route

    return best_route, best_distance
def calculate_route_demand(route, demands):
    total_demand = 0

    for node in route:
        total_demand += demands[node]

    return total_demand
def is_route_feasible(route, demands, vehicle_capacity):
    route_demand = calculate_route_demand(route, demands)

    return route_demand <= vehicle_capacity
def find_best_cvrp_bruteforce(
    num_customers,
    distance_matrix,
    demands,
    vehicle_capacity
):
    customer_nodes = list(range(1, num_customers + 1))

    best_routes = None
    best_total_distance = float("inf")

    for split_size in range(0, num_customers + 1):
        for vehicle1_group in itertools.combinations(
            customer_nodes,
            split_size
        ):
            vehicle1_customers = list(vehicle1_group)

            vehicle2_customers = [
                node
                for node in customer_nodes
                if node not in vehicle1_customers
            ]

            demand1 = sum(demands[node] for node in vehicle1_customers)
            demand2 = sum(demands[node] for node in vehicle2_customers)

            if demand1 > vehicle_capacity or demand2 > vehicle_capacity:
                continue

            for permutation1 in itertools.permutations(vehicle1_customers):
                route1 = [0] + list(permutation1) + [0]
                distance1 = calculate_route_distance(
                    route1,
                    distance_matrix
                )

                for permutation2 in itertools.permutations(vehicle2_customers):
                    route2 = [0] + list(permutation2) + [0]
                    distance2 = calculate_route_distance(
                        route2,
                        distance_matrix
                    )

                    total_distance = distance1 + distance2

                    if total_distance < best_total_distance:
                        best_total_distance = total_distance
                        best_routes = [route1, route2]

    return best_routes, best_total_distance
