from src.models import CVRPInstance, Solution


def calculate_optimality_gap(
    solution: Solution,
    reference_solution: Solution
) -> float | None:
    if not solution.feasible:
        return None

    if not reference_solution.feasible:
        return None

    if reference_solution.status != "OPTIMAL":
        raise ValueError(
            "Reference solution must be optimal."
        )

    if reference_solution.total_distance <= 0:
        raise ValueError(
            "Reference objective must be positive."
        )

    gap = (
        (
            solution.total_distance
            - reference_solution.total_distance
        )
        / reference_solution.total_distance
    ) * 100

    return gap


def compare_solvers(
    instance: CVRPInstance,
    solvers
):
    solutions = []

    for solver in solvers:
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

    comparison_rows = []

    for solution in solutions:
        if reference_solution is not None:
            gap = calculate_optimality_gap(
                solution,
                reference_solution
            )
        else:
            gap = None

        comparison_rows.append({
            "solver": solution.solver_name,
            "status": solution.status,
            "feasible": solution.feasible,
            "distance": solution.total_distance,
            "runtime_seconds":
                solution.runtime_seconds,
            "optimality_gap_percent": gap
        })

    return comparison_rows, solutions
