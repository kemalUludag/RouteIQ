import itertools
import time

from src.models import CVRPInstance, Solution
from src.distance import create_distance_matrix
from src.routing import calculate_route_distance
from src.validation import validate_cvrp_routes


class BruteForceCVRPSolver:
    def solve(
        self,
        instance: CVRPInstance
    ) -> Solution:
        start_time = time.perf_counter()

        if instance.num_vehicles != 2:
            runtime_seconds = (
                time.perf_counter()
                - start_time
            )

            return Solution(
                solver_name="Brute Force",
                routes=[],
                total_distance=float("inf"),
                runtime_seconds=runtime_seconds,
                feasible=False,
                status="UNSUPPORTED_INSTANCE",
                metadata={
                    "exact": True,
                    "reason":
                        "Current brute-force solver "
                        "supports exactly 2 vehicles."
                }
            )

        distance_matrix = (
            create_distance_matrix(
                instance.points
            )
        )

        depot = instance.depot_index

        customer_nodes = [
            node
            for node in range(instance.num_nodes)
            if node != depot
        ]

        capacity_1 = (
            instance.vehicle_capacities[0]
        )

        capacity_2 = (
            instance.vehicle_capacities[1]
        )

        best_routes = None
        best_total_distance = float("inf")

        candidate_solutions_evaluated = 0

        for split_size in range(
            len(customer_nodes) + 1
        ):
            for vehicle_1_group in itertools.combinations(
                customer_nodes,
                split_size
            ):
                vehicle_1_customers = list(
                    vehicle_1_group
                )

                vehicle_2_customers = [
                    node
                    for node in customer_nodes
                    if node not in vehicle_1_customers
                ]

                demand_1 = sum(
                    instance.demands[node]
                    for node in vehicle_1_customers
                )

                demand_2 = sum(
                    instance.demands[node]
                    for node in vehicle_2_customers
                )

                if demand_1 > capacity_1:
                    continue

                if demand_2 > capacity_2:
                    continue

                for permutation_1 in itertools.permutations(
                    vehicle_1_customers
                ):
                    route_1 = (
                        [depot]
                        + list(permutation_1)
                        + [depot]
                    )

                    distance_1 = (
                        calculate_route_distance(
                            route_1,
                            distance_matrix
                        )
                    )

                    for permutation_2 in itertools.permutations(
                        vehicle_2_customers
                    ):
                        route_2 = (
                            [depot]
                            + list(permutation_2)
                            + [depot]
                        )

                        distance_2 = (
                            calculate_route_distance(
                                route_2,
                                distance_matrix
                            )
                        )

                        candidate_solutions_evaluated += 1

                        total_distance = (
                            distance_1
                            + distance_2
                        )

                        if (
                            total_distance
                            < best_total_distance
                        ):
                            best_total_distance = (
                                total_distance
                            )

                            best_routes = [
                                route_1,
                                route_2
                            ]

        runtime_seconds = (
            time.perf_counter()
            - start_time
        )

        if best_routes is None:
            return Solution(
                solver_name="Brute Force",
                routes=[],
                total_distance=float("inf"),
                runtime_seconds=runtime_seconds,
                feasible=False,
                status="NO_SOLUTION",
                metadata={
                    "exact": True,
                    "candidate_solutions_evaluated":
                        candidate_solutions_evaluated
                }
            )

        is_valid, validation_message = (
            validate_cvrp_routes(
                instance,
                best_routes
            )
        )

        status = (
            "OPTIMAL"
            if is_valid
            else "INVALID_SOLUTION"
        )

        return Solution(
            solver_name="Brute Force",
            routes=best_routes,
            total_distance=best_total_distance,
            runtime_seconds=runtime_seconds,
            feasible=is_valid,
            status=status,
            metadata={
                "exact": True,
                "candidate_solutions_evaluated":
                    candidate_solutions_evaluated,
                "validation_message":
                    validation_message
            }
        )


    