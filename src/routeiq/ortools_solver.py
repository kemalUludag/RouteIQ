import time

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

from .validation import validate_cvrp_routes
from .models import CVRPInstance, Solution
from .distance import create_distance_matrix
from .routing import (
    calculate_route_distance,
    calculate_routes_cost
)


# ==================================================
# Core OR-Tools CVRP routing function
# ==================================================

def _solve_cvrp_routes(
    distance_matrix,
    demands,
    vehicle_capacities,
    time_limit_seconds=1
):
    num_nodes = len(
        distance_matrix
    )

    num_vehicles = len(
        vehicle_capacities
    )

    # ==========================================
    # OR-Tools index manager
    # ==========================================

    manager = pywrapcp.RoutingIndexManager(
        num_nodes,
        num_vehicles,
        0
    )

    routing = pywrapcp.RoutingModel(
        manager
    )

    # ==========================================
    # Arc-cost callback
    # ==========================================

    def distance_callback(
        from_index,
        to_index
    ):
        from_node = (
            manager.IndexToNode(
                from_index
            )
        )

        to_node = (
            manager.IndexToNode(
                to_index
            )
        )

        distance = (
            distance_matrix[
                from_node
            ][
                to_node
            ]
        )

        # OR-Tools routing costs must be integers.
        #
        # We multiply by 100 to preserve
        # approximately two decimal places.

        return int(
            round(
                distance * 100
            )
        )

    transit_callback_index = (
        routing.RegisterTransitCallback(
            distance_callback
        )
    )

    routing.SetArcCostEvaluatorOfAllVehicles(
        transit_callback_index
    )

    # ==========================================
    # Demand callback
    # ==========================================

    def demand_callback(
        from_index
    ):
        from_node = (
            manager.IndexToNode(
                from_index
            )
        )

        return demands[
            from_node
        ]

    demand_callback_index = (
        routing.RegisterUnaryTransitCallback(
            demand_callback
        )
    )

    # ==========================================
    # Capacity constraints
    # ==========================================

    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        vehicle_capacities,
        True,
        "Capacity"
    )

    # ==========================================
    # Search configuration
    # ==========================================

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

    # ==========================================
    # Solve
    # ==========================================

    solution = (
        routing.SolveWithParameters(
            search_parameters
        )
    )

    if solution is None:
        return None

    # ==========================================
    # Extract routes
    # ==========================================

    routes = []

    for vehicle_id in range(
        num_vehicles
    ):
        route = []

        index = routing.Start(
            vehicle_id
        )

        while not routing.IsEnd(
            index
        ):
            node = (
                manager.IndexToNode(
                    index
                )
            )

            route.append(
                node
            )

            index = (
                solution.Value(
                    routing.NextVar(
                        index
                    )
                )
            )

        end_node = (
            manager.IndexToNode(
                index
            )
        )

        route.append(
            end_node
        )

        routes.append(
            route
        )

    return routes


# ==================================================
# Legacy functional API
# ==================================================

def solve_cvrp_ortools(
    distance_matrix,
    demands,
    num_vehicles,
    vehicle_capacity,
    time_limit_seconds=1
):
    vehicle_capacities = [
        vehicle_capacity
        for _ in range(
            num_vehicles
        )
    ]

    return _solve_cvrp_routes(
        distance_matrix=distance_matrix,
        demands=demands,
        vehicle_capacities=vehicle_capacities,
        time_limit_seconds=time_limit_seconds
    )


# ==================================================
# RouteIQ OR-Tools solver wrapper
# ==================================================

class ORToolsCVRPSolver:

    def __init__(
        self,
        time_limit_seconds=1,
        solver_name=None
    ):
        self.time_limit_seconds = (
            time_limit_seconds
        )

        if solver_name is None:
            self.solver_name = (
                "OR-Tools"
            )
        else:
            self.solver_name = (
                solver_name
            )

    # ==============================================
    # Solver capability
    # ==============================================

    def supports(
        self,
        instance: CVRPInstance
    ) -> bool:
        return True

    # ==============================================
    # Standard distance-based solve
    # ==============================================

    def solve(
        self,
        instance: CVRPInstance
    ) -> Solution:

        distance_matrix = (
            create_distance_matrix(
                instance.points
            )
        )

        start_time = (
            time.perf_counter()
        )

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

        # ==========================================
        # No solution
        # ==========================================

        if routes is None:
            return Solution(
                solver_name=self.solver_name,
                routes=[],
                total_distance=float(
                    "inf"
                ),
                runtime_seconds=(
                    runtime_seconds
                ),
                feasible=False,
                status="NO_SOLUTION",
                metadata={
                    "time_limit_seconds":
                        self.time_limit_seconds
                }
            )

        # ==========================================
        # Calculate Euclidean objective
        # ==========================================

        total_distance = sum(
            calculate_route_distance(
                route,
                distance_matrix
            )
            for route in routes
        )

        # ==========================================
        # Independent RouteIQ validation
        # ==========================================

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

        # ==========================================
        # Return RouteIQ solution
        # ==========================================

        return Solution(
            solver_name=self.solver_name,
            routes=routes,
            total_distance=total_distance,
            runtime_seconds=runtime_seconds,
            feasible=is_valid,
            status=status,
            metadata={
                "time_limit_seconds":
                    self.time_limit_seconds,

                "validation_message":
                    validation_message,

                "objective_name":
                    "distance",

                "objective_unit":
                    "distance_units",

                "custom_cost_matrix":
                    False
            }
        )

    # ==============================================
    # Generic custom-cost solve
    # ==============================================

    def solve_with_cost_matrix(
        self,
        instance: CVRPInstance,
        cost_matrix,
        objective_name="custom_cost",
        objective_unit="units"
    ) -> Solution:

        start_time = (
            time.perf_counter()
        )

        # ==========================================
        # Basic matrix validation
        # ==========================================

        expected_size = (
            instance.num_nodes
        )

        if (
            len(cost_matrix)
            != expected_size
        ):
            raise ValueError(
                "Cost matrix row count must match "
                "the number of instance nodes."
            )

        for row in cost_matrix:
            if (
                len(row)
                != expected_size
            ):
                raise ValueError(
                    "Cost matrix must be square and "
                    "match the number of instance nodes."
                )

        # ==========================================
        # Run existing OR-Tools routing core
        #
        # IMPORTANT:
        # Although the core parameter is still
        # called distance_matrix for backward
        # compatibility, it can contain any
        # non-negative arc-cost matrix.
        # ==========================================

        routes = _solve_cvrp_routes(
            distance_matrix=cost_matrix,
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

        # ==========================================
        # No solution
        # ==========================================

        if routes is None:
            return Solution(
                solver_name=self.solver_name,
                routes=[],
                total_distance=float(
                    "inf"
                ),
                runtime_seconds=(
                    runtime_seconds
                ),
                feasible=False,
                status="NO_SOLUTION",
                metadata={
                    "time_limit_seconds":
                        self.time_limit_seconds,

                    "objective_name":
                        objective_name,

                    "objective_unit":
                        objective_unit,

                    "custom_cost_matrix":
                        True
                }
            )

        # ==========================================
        # Independent RouteIQ validation
        # ==========================================

        is_valid, validation_message = (
            validate_cvrp_routes(
                instance,
                routes
            )
        )

        # ==========================================
        # Calculate objective using original
        # floating-point cost matrix
        # ==========================================

        total_cost = (
            calculate_routes_cost(
                routes,
                cost_matrix
            )
        )

        status = (
            "FEASIBLE"
            if is_valid
            else "INVALID_SOLUTION"
        )

        # ==========================================
        # Return standard RouteIQ Solution
        #
        # NOTE:
        # total_distance temporarily stores the
        # generic objective value.
        #
        # We retain this for compatibility with
        # the current Solution dataclass.
        # ==========================================

        return Solution(
            solver_name=self.solver_name,
            routes=routes,
            total_distance=total_cost,
            runtime_seconds=runtime_seconds,
            feasible=is_valid,
            status=status,
            metadata={
                "time_limit_seconds":
                    self.time_limit_seconds,

                "validation_message":
                    validation_message,

                "objective_name":
                    objective_name,

                "objective_unit":
                    objective_unit,

                "custom_cost_matrix":
                    True
            }
        )
    