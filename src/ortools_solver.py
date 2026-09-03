from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

def solve_cvrp_ortools(
    distance_matrix,
    demands,
    num_vehicles,
    vehicle_capacity,
    time_limit_seconds=1
):
    num_nodes = len(distance_matrix)

    manager = pywrapcp.RoutingIndexManager(
        num_nodes,
        num_vehicles,
        0
    )

    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)

        distance = distance_matrix[from_node][to_node]

        return int(round(distance * 100))

    transit_callback_index = routing.RegisterTransitCallback(
        distance_callback
    )

    routing.SetArcCostEvaluatorOfAllVehicles(
        transit_callback_index
    )

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)

        return demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback
    )

    vehicle_capacities = [
        vehicle_capacity
        for _ in range(num_vehicles)
    ]

    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        vehicle_capacities,
        True,
        "Capacity"
    )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )

    search_parameters.time_limit.FromSeconds(
    time_limit_seconds
   )

    solution = routing.SolveWithParameters(search_parameters)

    if solution is None:
        return None

    routes = []

    for vehicle_id in range(num_vehicles):
        route = []

        index = routing.Start(vehicle_id)

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)

            index = solution.Value(
                routing.NextVar(index)
            )

        end_node = manager.IndexToNode(index)
        route.append(end_node)

        routes.append(route)

    return routes
