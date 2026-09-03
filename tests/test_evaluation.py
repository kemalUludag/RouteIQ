import pytest

from src.models import Solution
from src.evaluation import (
    calculate_optimality_gap
)


def test_zero_optimality_gap():
    reference = Solution(
        solver_name="Exact Solver",
        routes=[],
        total_distance=100.0,
        runtime_seconds=1.0,
        feasible=True,
        status="OPTIMAL"
    )

    solution = Solution(
        solver_name="Test Solver",
        routes=[],
        total_distance=100.0,
        runtime_seconds=0.5,
        feasible=True,
        status="FEASIBLE"
    )

    gap = calculate_optimality_gap(
        solution,
        reference
    )

    assert gap == pytest.approx(0.0)


def test_positive_optimality_gap():
    reference = Solution(
        solver_name="Exact Solver",
        routes=[],
        total_distance=100.0,
        runtime_seconds=1.0,
        feasible=True,
        status="OPTIMAL"
    )

    solution = Solution(
        solver_name="Test Solver",
        routes=[],
        total_distance=105.0,
        runtime_seconds=0.5,
        feasible=True,
        status="FEASIBLE"
    )

    gap = calculate_optimality_gap(
        solution,
        reference
    )

    assert gap == pytest.approx(5.0)


def test_infeasible_solution_has_no_gap():
    reference = Solution(
        solver_name="Exact Solver",
        routes=[],
        total_distance=100.0,
        runtime_seconds=1.0,
        feasible=True,
        status="OPTIMAL"
    )

    solution = Solution(
        solver_name="Test Solver",
        routes=[],
        total_distance=float("inf"),
        runtime_seconds=0.5,
        feasible=False,
        status="NO_SOLUTION"
    )

    gap = calculate_optimality_gap(
        solution,
        reference
    )

    assert gap is None

    