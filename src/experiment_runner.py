import pandas as pd

from src.evaluation import calculate_optimality_gap


class ExperimentRunner:
    def __init__(self, solvers):
        self.solvers = solvers

    def run(self, instances):
        records = []

        for instance in instances:
            solutions = []

            for solver in self.solvers:
                if not solver.supports(instance):
                    continue

                solution = solver.solve(instance)
                solutions.append(solution)

            reference_solution = None

            for solution in solutions:
                if (
                    solution.feasible
                    and solution.status == "OPTIMAL"
                ):
                    reference_solution = solution
                    break

            for solution in solutions:
                if reference_solution is not None:
                    gap = calculate_optimality_gap(
                        solution,
                        reference_solution
                    )
                else:
                    gap = None

                records.append({
                    "instance": instance.name,
                    "num_customers":
                        instance.num_customers,
                    "num_vehicles":
                        instance.num_vehicles,
                    "solver":
                        solution.solver_name,
                    "status":
                        solution.status,
                    "feasible":
                        solution.feasible,
                    "distance":
                        solution.total_distance,
                    "runtime_seconds":
                        solution.runtime_seconds,
                    "optimality_gap_percent":
                        gap
                })

        return pd.DataFrame(records)

    