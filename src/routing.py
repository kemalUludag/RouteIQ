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
