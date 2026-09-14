import time

import gurobipy as gp
from gurobipy import GRB

from .models import (
    CVRPInstance,
    Solution
)

from .distance import (
    create_distance_matrix
)

from .routing import (
    calculate_routes_cost
)

from .validation import (
    validate_cvrp_routes
)


class GurobiCVRPSolver:

    def __init__(
        self,
        time_limit_seconds=30,
        output_flag=0,
        max_customers=None
    ):
        self.time_limit_seconds = (
            time_limit_seconds
        )

        self.output_flag = (
            output_flag
        )

        self.max_customers = (
            max_customers
        )

    # ==================================================
    # Solver capability
    # ==================================================

    def supports(
        self,
        instance: CVRPInstance
    ) -> bool:

        if self.max_customers is None:
            return True

        return (
            instance.num_customers
            <= self.max_customers
        )

    # ==================================================
    # Cost-matrix validation
    # ==================================================

    def _validate_cost_matrix(
        self,
        instance: CVRPInstance,
        cost_matrix
    ):

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

    # ==================================================
    # Standard distance-based interface
    # ==================================================

    def solve(
        self,
        instance: CVRPInstance
    ) -> Solution:

        distance_matrix = (
            create_distance_matrix(
                instance.points
            )
        )

        return self._solve_with_cost_matrix(
            instance=instance,
            cost_matrix=distance_matrix,
            objective_name="distance",
            objective_unit="distance_units"
        )

    # ==================================================
    # Public custom-cost interface
    # ==================================================

    def solve_with_cost_matrix(
        self,
        instance: CVRPInstance,
        cost_matrix,
        objective_name="custom_cost",
        objective_unit="units"
    ) -> Solution:

        self._validate_cost_matrix(
            instance,
            cost_matrix
        )

        return self._solve_with_cost_matrix(
            instance=instance,
            cost_matrix=cost_matrix,
            objective_name=objective_name,
            objective_unit=objective_unit
        )

    # ==================================================
    # Generic MILP implementation
    # ==================================================

    def _solve_with_cost_matrix(
        self,
        instance: CVRPInstance,
        cost_matrix,
        objective_name,
        objective_unit
    ) -> Solution:

        start_time = (
            time.perf_counter()
        )

        # ==============================================
        # Applicability check
        # ==============================================

        if not self.supports(
            instance
        ):
            return Solution(
                solver_name="Gurobi-MILP",
                routes=[],
                total_distance=float(
                    "inf"
                ),
                runtime_seconds=(
                    time.perf_counter()
                    - start_time
                ),
                feasible=False,
                status="UNSUPPORTED_INSTANCE",
                metadata={
                    "max_customers":
                        self.max_customers,

                    "objective_name":
                        objective_name,

                    "objective_unit":
                        objective_unit
                }
            )

        # ==============================================
        # Problem sets
        # ==============================================

        depot = (
            instance.depot_index
        )

        nodes = list(
            range(
                instance.num_nodes
            )
        )

        customers = [
            node
            for node in nodes
            if node != depot
        ]

        vehicles = list(
            range(
                instance.num_vehicles
            )
        )

        # ==============================================
        # Create optimization model
        # ==============================================

        model = gp.Model(
            "RouteIQ_CVRP"
        )

        model.Params.OutputFlag = (
            self.output_flag
        )

        model.Params.TimeLimit = (
            self.time_limit_seconds
        )

        # ==============================================
        # Decision variables
        # ==============================================

        # x[i,j,k] = 1 if vehicle k
        # travels directly from node i to node j

        x = model.addVars(
            [
                (
                    i,
                    j,
                    k
                )
                for i in nodes
                for j in nodes
                for k in vehicles
                if i != j
            ],
            vtype=GRB.BINARY,
            name="x"
        )

        # y[i,k] = 1 if customer i
        # is assigned to vehicle k

        y = model.addVars(
            [
                (
                    i,
                    k
                )
                for i in customers
                for k in vehicles
            ],
            vtype=GRB.BINARY,
            name="y"
        )

        # vehicle_used[k] = 1 if vehicle k
        # is actually used

        vehicle_used = model.addVars(
            vehicles,
            vtype=GRB.BINARY,
            name="vehicle_used"
        )

        # load[i,k] =
        # cumulative delivered load when
        # vehicle k reaches customer i

        load = model.addVars(
            [
                (
                    i,
                    k
                )
                for i in customers
                for k in vehicles
            ],
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            name="load"
        )

        # ==============================================
        # Objective
        # ==============================================

        model.setObjective(
            gp.quicksum(
                cost_matrix[i][j]
                * x[i, j, k]

                for i in nodes
                for j in nodes
                for k in vehicles

                if i != j
            ),
            GRB.MINIMIZE
        )

        # ==============================================
        # Every customer is served exactly once
        # ==============================================

        for i in customers:

            model.addConstr(
                gp.quicksum(
                    y[i, k]
                    for k in vehicles
                )
                == 1,

                name=f"assignment_{i}"
            )

        # ==============================================
        # Customer flow conservation
        # ==============================================

        for i in customers:

            for k in vehicles:

                # If customer i belongs to vehicle k,
                # exactly one arc must enter i.

                model.addConstr(
                    gp.quicksum(
                        x[j, i, k]

                        for j in nodes

                        if j != i
                    )
                    == y[i, k],

                    name=(
                        f"inflow_{i}_{k}"
                    )
                )

                # And exactly one arc must leave i.

                model.addConstr(
                    gp.quicksum(
                        x[i, j, k]

                        for j in nodes

                        if j != i
                    )
                    == y[i, k],

                    name=(
                        f"outflow_{i}_{k}"
                    )
                )

        # ==============================================
        # Depot usage
        # ==============================================

        for k in vehicles:

            # Active vehicle leaves depot once.

            model.addConstr(
                gp.quicksum(
                    x[depot, j, k]

                    for j in customers
                )
                == vehicle_used[k],

                name=(
                    f"depot_departure_{k}"
                )
            )

            # Active vehicle returns once.

            model.addConstr(
                gp.quicksum(
                    x[i, depot, k]

                    for i in customers
                )
                == vehicle_used[k],

                name=(
                    f"depot_return_{k}"
                )
            )

        # ==============================================
        # Customer assignment requires active vehicle
        # ==============================================

        for i in customers:

            for k in vehicles:

                model.addConstr(
                    y[i, k]
                    <= vehicle_used[k],

                    name=(
                        f"vehicle_link_{i}_{k}"
                    )
                )

        # ==============================================
        # Vehicle capacity
        # ==============================================

        for k in vehicles:

            model.addConstr(
                gp.quicksum(
                    instance.demands[i]
                    * y[i, k]

                    for i in customers
                )
                <= (
                    instance.vehicle_capacities[k]
                    * vehicle_used[k]
                ),

                name=(
                    f"capacity_{k}"
                )
            )

        # ==============================================
        # Load bounds
        # ==============================================

        for i in customers:

            for k in vehicles:

                model.addConstr(
                    load[i, k]
                    >= (
                        instance.demands[i]
                        * y[i, k]
                    ),

                    name=(
                        f"load_lower_{i}_{k}"
                    )
                )

                model.addConstr(
                    load[i, k]
                    <= (
                        instance.vehicle_capacities[k]
                        * y[i, k]
                    ),

                    name=(
                        f"load_upper_{i}_{k}"
                    )
                )

        # ==============================================
        # Load propagation + subtour elimination
        # ==============================================

        max_customer_demand = max(
            instance.demands[i]
            for i in customers
        )

        for k in vehicles:

            capacity = (
                instance.vehicle_capacities[k]
            )

            # Big-M must fully deactivate the
            # constraint when arc i -> j is unused.

            big_m = (
                capacity
                + max_customer_demand
            )

            for i in customers:

                for j in customers:

                    if i == j:
                        continue

                    model.addConstr(
                        load[j, k]
                        >= (
                            load[i, k]
                            + instance.demands[j]

                            - big_m
                            * (
                                1
                                - x[i, j, k]
                            )
                        ),

                        name=(
                            f"load_flow_"
                            f"{i}_{j}_{k}"
                        )
                    )

        # ==============================================
        # Finalize and optimize
        # ==============================================

        model.update()

        model.optimize()

        runtime_seconds = (
            time.perf_counter()
            - start_time
        )

        # ==============================================
        # No incumbent solution
        # ==============================================

        if model.SolCount == 0:

            if (
                model.Status
                == GRB.INFEASIBLE
            ):
                status = (
                    "INFEASIBLE"
                )

            else:
                status = (
                    "NO_SOLUTION"
                )

            return Solution(
                solver_name="Gurobi-MILP",
                routes=[],
                total_distance=float(
                    "inf"
                ),
                runtime_seconds=(
                    runtime_seconds
                ),
                feasible=False,
                status=status,
                metadata={
                    "gurobi_status":
                        model.Status,

                    "solution_count":
                        model.SolCount,

                    "time_limit_seconds":
                        self.time_limit_seconds,

                    "objective_name":
                        objective_name,

                    "objective_unit":
                        objective_unit,

                    "custom_cost_matrix":
                        (
                            objective_name
                            != "distance"
                        )
                }
            )

        # ==============================================
        # Extract routes
        # ==============================================

        routes = []

        for k in vehicles:

            if (
                vehicle_used[k].X
                < 0.5
            ):
                routes.append(
                    [
                        depot,
                        depot
                    ]
                )

                continue

            route = [
                depot
            ]

            current_node = (
                depot
            )

            max_steps = (
                instance.num_nodes
                + 1
            )

            steps = 0

            while (
                steps
                < max_steps
            ):

                next_node = None

                for j in nodes:

                    if (
                        j
                        == current_node
                    ):
                        continue

                    key = (
                        current_node,
                        j,
                        k
                    )

                    if (
                        key in x
                        and x[key].X > 0.5
                    ):
                        next_node = j
                        break

                if next_node is None:
                    break

                route.append(
                    next_node
                )

                current_node = (
                    next_node
                )

                steps += 1

                if (
                    current_node
                    == depot
                ):
                    break

            routes.append(
                route
            )

        # ==============================================
        # Independent RouteIQ validation
        # ==============================================

        is_valid, validation_message = (
            validate_cvrp_routes(
                instance,
                routes
            )
        )

        # ==============================================
        # Recalculate objective using original
        # floating-point matrix
        # ==============================================

        total_cost = (
            calculate_routes_cost(
                routes,
                cost_matrix
            )
        )

        # ==============================================
        # Translate optimization status
        # ==============================================

        if (
            model.Status
            == GRB.OPTIMAL
            and is_valid
        ):
            status = (
                "OPTIMAL"
            )

        elif is_valid:
            status = (
                "FEASIBLE"
            )

        else:
            status = (
                "INVALID_SOLUTION"
            )

        # ==============================================
        # Metadata
        # ==============================================

        metadata = {
            "gurobi_status":
                model.Status,

            "solution_count":
                model.SolCount,

            "node_count":
                model.NodeCount,

            "best_bound":
                model.ObjBound,

            "time_limit_seconds":
                self.time_limit_seconds,

            "validation_message":
                validation_message,

            "objective_name":
                objective_name,

            "objective_unit":
                objective_unit,

            "custom_cost_matrix":
                (
                    objective_name
                    != "distance"
                )
        }

        if model.IsMIP:

            metadata[
                "mip_gap"
            ] = (
                model.MIPGap
            )

        # ==============================================
        # Standard RouteIQ solution
        # ==============================================

        return Solution(
            solver_name="Gurobi-MILP",
            routes=routes,
            total_distance=total_cost,
            runtime_seconds=runtime_seconds,
            feasible=is_valid,
            status=status,
            metadata=metadata
        )

    