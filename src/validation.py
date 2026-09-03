from src.models import CVRPInstance



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


def validate_cvrp_routes(
    instance: CVRPInstance,
    routes: list[list[int]]
) -> tuple[bool, str]:

    if len(routes) != instance.num_vehicles:
        return (
            False,
            "Incorrect number of vehicle routes."
        )

    valid_nodes = set(
        range(instance.num_nodes)
    )

    visited_customers = []

    for vehicle_id, route in enumerate(routes):
        if not route:
            return (
                False,
                f"Vehicle {vehicle_id} has an empty route."
            )

        if (
            route[0] != instance.depot_index
            or route[-1] != instance.depot_index
        ):
            return (
                False,
                f"Vehicle {vehicle_id} must start and end at the depot."
            )

        for node in route:
            if node not in valid_nodes:
                return (
                    False,
                    f"Route contains invalid node: {node}."
                )

        if instance.depot_index in route[1:-1]:
            return (
                False,
                "Depot cannot appear inside a route."
            )

        route_demand = sum(
            instance.demands[node]
            for node in route
        )

        vehicle_capacity = (
            instance.vehicle_capacities[
                vehicle_id
            ]
        )

        if route_demand > vehicle_capacity:
            return (
                False,
                f"Vehicle {vehicle_id} capacity is violated."
            )

        for node in route[1:-1]:
            visited_customers.append(node)

    expected_customers = [
        node
        for node in range(instance.num_nodes)
        if node != instance.depot_index
    ]

    if (
        sorted(visited_customers)
        != sorted(expected_customers)
    ):
        return (
            False,
            "Customers are missing or visited more than once."
        )

    return (
        True,
        "Solution is feasible."
    )

