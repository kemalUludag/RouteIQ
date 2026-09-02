from src.routing import calculate_route_distance, calculate_route_demand


def print_solution_report(
    routes,
    distance_matrix,
    demands,
    vehicle_capacity
):
    total_distance = 0

    for vehicle_id, route in enumerate(routes, start=1):
        route_distance = calculate_route_distance(
            route,
            distance_matrix
        )

        route_demand = calculate_route_demand(
            route,
            demands
        )

        capacity_utilization = (
            route_demand / vehicle_capacity
        ) * 100

        total_distance += route_distance

        print(f"Vehicle {vehicle_id}")
        print(f"  Route: {route}")
        print(f"  Load: {route_demand}/{vehicle_capacity}")
        print(f"  Capacity utilization: {capacity_utilization:.1f}%")
        print(f"  Distance: {route_distance:.2f}")
        print()

    print(f"Total distance: {total_distance:.2f}")
    