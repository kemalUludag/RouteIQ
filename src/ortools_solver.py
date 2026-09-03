import time

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

from src.validation import validate_cvrp_routes
from src.models import CVRPInstance, Solution
from src.distance import create_distance_matrix
from src.routing import calculate_route_distance


def _solve_cvrp_routes(
    distance_matrix,
    demands,
    vehicle_capacities,
    time_limit_seconds=1
):
    num_nodes = len(distance_matrix)
    num_vehicles = len(vehicle_capacities)

    manager = pywrapcp.RoutingIndexManager(
        num_nodes,
        num_vehicles,
        0
    )

    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(
        from_index,
        to_index
    ):
        from_node = manager.IndexToNode(
            from_index
        )

        to_node = manager.IndexToNode(
            to_index
        )

        distance = distance_matrix[
            from_node
        ][
            to_node
        ]

        return int(
            round(distance * 100)
        )

    transit_callback_index = (
        routing.RegisterTransitCallback(
            distance_callback
        )
    )

    routing.SetArcCostEvaluatorOfAllVehicles(
        transit_callback_index
    )

    def demand_callback(from_index):
        from_node = manager.IndexToNode(
            from_index
        )

        return demands[from_node]

    demand_callback_index = (
        routing.RegisterUnaryTransitCallback(
            demand_callback
        )
    )

    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        vehicle_capacities,
        True,
        "Capacity"
    )

    search_parameters = (
        pywrapcp.DefaultRoutingSearchParameters()
    )

    search_parameters.first_solution_strategy = (
        routing_enums_pb2
        .FirstSolutionStrategy
        .PATH_CHEAPEST_ARC
    )

    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2
        .LocalSearchMetaheuristic
        .GUIDED_LOCAL_SEARCH
    )

    search_parameters.time_limit.FromSeconds(
        time_limit_seconds
    )

    solution = routing.SolveWithParameters(
        search_parameters
    )

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


def solve_cvrp_ortools(
    distance_matrix,
    demands,
    num_vehicles,
    vehicle_capacity,
    time_limit_seconds=1
):
    vehicle_capacities = [
        vehicle_capacity
        for _ in range(num_vehicles)
    ]

    return _solve_cvrp_routes(
        distance_matrix=distance_matrix,
        demands=demands,
        vehicle_capacities=vehicle_capacities,
        time_limit_seconds=time_limit_seconds
    )


class ORToolsCVRPSolver:
    def __init__(
        self,
        time_limit_seconds=1
    ):
        self.time_limit_seconds = (
            time_limit_seconds
        )

    def solve(
        self,
        instance: CVRPInstance
    ) -> Solution:
        distance_matrix = (
            create_distance_matrix(
                instance.points
            )
        )

        start_time = time.perf_counter()

        routes = _solve_cvrp_routes(
            distance_matrix=distance_matrix,
            demands=instance.demands,
            vehicle_capacities=(
                instance.vehicle_capacities
            ),
            time_limit_seconds=(
                self.time_limit_seconds
            )
        )

        runtime_seconds = (
            time.perf_counter()
            - start_time
        )

        if routes is None:
            return Solution(
                solver_name="OR-Tools",
                routes=[],
                total_distance=float("inf"),
                runtime_seconds=runtime_seconds,
                feasible=False,
                status="NO_SOLUTION",
                metadata={
                    "time_limit_seconds":
                        self.time_limit_seconds
                }
            )

        total_distance = sum(
            calculate_route_distance(
                route,
                distance_matrix
            )
            for route in routes
        )

        is_valid, validation_message = (
            validate_cvrp_routes(
                instance,
                routes
            )
        )

        status = (
            "FEASIBLE"
            if is_valid
            else "INVALID_SOLUTION"
        )

        return Solution(
            solver_name="OR-Tools",
            routes=routes,
            total_distance=total_distance,
            runtime_seconds=runtime_seconds,
            feasible=is_valid,
            status=status,
            metadata={
                "time_limit_seconds":
                    self.time_limit_seconds,
                "validation_message":
                    validation_message
            }
        )

    