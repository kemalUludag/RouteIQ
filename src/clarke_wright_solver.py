import time

from src.models import CVRPInstance, Solution
from src.distance import create_distance_matrix
from src.routing import calculate_route_distance
from src.validation import validate_cvrp_routes


class ClarkeWrightCVRPSolver:
    def supports(
        self,
        instance: CVRPInstance
    ) -> bool:
        return (
            len(
                set(instance.vehicle_capacities)
            )
            == 1
        )

    def solve(
        self,
        instance: CVRPInstance
    ) -> Solution:
        start_time = time.perf_counter()

        if not self.supports(instance):
            return Solution(
                solver_name="Clarke-Wright",
                routes=[],
                total_distance=float("inf"),
                runtime_seconds=(
                    time.perf_counter()
                    - start_time
                ),
                feasible=False,
                status="UNSUPPORTED_INSTANCE",
                metadata={
                    "reason":
                        "Current Clarke-Wright "
                        "implementation requires "
                        "homogeneous vehicle capacities."
                }
            )

        distance_matrix = (
            create_distance_matrix(
                instance.points
            )
        )

        depot = instance.depot_index

        customers = [
            node
            for node in range(instance.num_nodes)
            if node != depot
        ]

        vehicle_capacity = (
            instance.vehicle_capacities[0]
        )

        routes = {
            customer: [
                depot,
                customer,
                depot
            ]
            for customer in customers
        }

        route_loads = {
            customer:
                instance.demands[customer]
            for customer in customers
        }

        customer_to_route = {
            customer: customer
            for customer in customers
        }

        savings = []

        for i_index in range(len(customers)):
            for j_index in range(
                i_index + 1,
                len(customers)
            ):
                i = customers[i_index]
                j = customers[j_index]

                saving = (
                    distance_matrix[depot][i]
                    + distance_matrix[depot][j]
                    - distance_matrix[i][j]
                )

                savings.append(
                    (
                        saving,
                        i,
                        j
                    )
                )

        savings.sort(
            reverse=True
        )

        merges_performed = 0

        for saving, i, j in savings:
            route_id_i = (
                customer_to_route[i]
            )

            route_id_j = (
                customer_to_route[j]
            )

            if route_id_i == route_id_j:
                continue

            route_i = routes[route_id_i]
            route_j = routes[route_id_j]

            combined_load = (
                route_loads[route_id_i]
                + route_loads[route_id_j]
            )

            if combined_load > vehicle_capacity:
                continue

            merged_route = None

            if (
                route_i[-2] == i
                and route_j[1] == j
            ):
                merged_route = (
                    route_i[:-1]
                    + route_j[1:]
                )

            elif (
                route_i[1] == i
                and route_j[-2] == j
            ):
                merged_route = (
                    route_j[:-1]
                    + route_i[1:]
                )

            elif (
                route_i[1] == i
                and route_j[1] == j
            ):
                reversed_route_i = (
                    [depot]
                    + list(
                        reversed(
                            route_i[1:-1]
                        )
                    )
                    + [depot]
                )

                merged_route = (
                    reversed_route_i[:-1]
                    + route_j[1:]
                )

            elif (
                route_i[-2] == i
                and route_j[-2] == j
            ):
                reversed_route_j = (
                    [depot]
                    + list(
                        reversed(
                            route_j[1:-1]
                        )
                    )
                    + [depot]
                )

                merged_route = (
                    route_i[:-1]
                    + reversed_route_j[1:]
                )

            if merged_route is None:
                continue

            new_route_id = route_id_i

            routes[new_route_id] = (
                merged_route
            )

            route_loads[new_route_id] = (
                combined_load
            )

            del routes[route_id_j]
            del route_loads[route_id_j]

            for customer in merged_route[1:-1]:
                customer_to_route[
                    customer
                ] = new_route_id

            merges_performed += 1

        final_routes = list(
            routes.values()
        )

        if (
            len(final_routes)
            > instance.num_vehicles
        ):
            runtime_seconds = (
                time.perf_counter()
                - start_time
            )

            return Solution(
                solver_name="Clarke-Wright",
                routes=final_routes,
                total_distance=float("inf"),
                runtime_seconds=runtime_seconds,
                feasible=False,
                status="NO_SOLUTION",
                metadata={
                    "merges_performed":
                        merges_performed,
                    "reason":
                        "Heuristic produced more "
                        "routes than available vehicles."
                }
            )

        while (
            len(final_routes)
            < instance.num_vehicles
        ):
            final_routes.append(
                [depot, depot]
            )

        total_distance = sum(
            calculate_route_distance(
                route,
                distance_matrix
            )
            for route in final_routes
        )

        is_valid, validation_message = (
            validate_cvrp_routes(
                instance,
                final_routes
            )
        )

        runtime_seconds = (
            time.perf_counter()
            - start_time
        )

        status = (
            "FEASIBLE"
            if is_valid
            else "INVALID_SOLUTION"
        )

        return Solution(
            solver_name="Clarke-Wright",
            routes=final_routes,
            total_distance=total_distance,
            runtime_seconds=runtime_seconds,
            feasible=is_valid,
            status=status,
            metadata={
                "merges_performed":
                    merges_performed,
                "validation_message":
                    validation_message
            }
        )

    