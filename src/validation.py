def validate_solution(
    routes,
    num_customers,
    demands,
    vehicle_capacity,
    num_vehicles
):
    if len(routes) != num_vehicles:
        return False, "Incorrect number of vehicle routes."

    visited_customers = []

    for route in routes:
        if route[0] != 0 or route[-1] != 0:
            return False, "Every route must start and end at the depot."

        route_demand = sum(demands[node] for node in route)

        if route_demand > vehicle_capacity:
            return False, "A vehicle capacity constraint is violated."

        for node in route[1:-1]:
            visited_customers.append(node)

    expected_customers = list(range(1, num_customers + 1))

    if sorted(visited_customers) != expected_customers:
        return False, "Customers are missing or visited more than once."

    return True, "Solution is feasible."
