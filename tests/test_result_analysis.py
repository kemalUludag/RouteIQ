import pandas as pd
import pytest

from src.result_analysis import (
    compare_solver_to_baseline,
    bootstrap_mean_confidence_interval
)


def test_compare_solver_to_baseline():
    results_df = pd.DataFrame([
        {
            "instance": "test_1",
            "num_customers": 10,
            "solver": "Baseline",
            "distance": 100.0,
            "runtime_seconds": 1.0,
            "feasible": True
        },
        {
            "instance": "test_1",
            "num_customers": 10,
            "solver": "Challenger",
            "distance": 90.0,
            "runtime_seconds": 2.0,
            "feasible": True
        }
    ])

    comparison_df = (
        compare_solver_to_baseline(
            results_df,
            baseline_solver="Baseline",
            challenger_solver="Challenger"
        )
    )

    row = comparison_df.iloc[0]

    assert (
        row["objective_improvement_percent"]
        == pytest.approx(10.0)
    )

    assert (
        row["runtime_ratio"]
        == pytest.approx(2.0)
    )

def test_challenger_can_be_worse_than_baseline():
    results_df = pd.DataFrame([
        {
            "instance": "test_1",
            "num_customers": 10,
            "solver": "Baseline",
            "distance": 100.0,
            "runtime_seconds": 1.0,
            "feasible": True
        },
        {
            "instance": "test_1",
            "num_customers": 10,
            "solver": "Challenger",
            "distance": 105.0,
            "runtime_seconds": 2.0,
            "feasible": True
        }
    ])

    comparison_df = (
        compare_solver_to_baseline(
            results_df,
            baseline_solver="Baseline",
            challenger_solver="Challenger"
        )
    )

    row = comparison_df.iloc[0]

    assert (
        row["objective_improvement_percent"]
        == pytest.approx(-5.0)
    )

    assert row["outcome"] == "loss"


def test_infeasible_baseline_is_excluded_from_pairwise_comparison():
    results_df = pd.DataFrame([
        {
            "instance": "test_1",
            "num_customers": 20,
            "solver": "Baseline",
            "distance": float("inf"),
            "runtime_seconds": 0.1,
            "feasible": False
        },
        {
            "instance": "test_1",
            "num_customers": 20,
            "solver": "Challenger",
            "distance": 100.0,
            "runtime_seconds": 1.0,
            "feasible": True
        }
    ])

    comparison_df = (
        compare_solver_to_baseline(
            results_df,
            baseline_solver="Baseline",
            challenger_solver="Challenger"
        )
    )

    assert comparison_df.empty




def test_bootstrap_confidence_interval_for_constant_values():
    values = [
        5.0,
        5.0,
        5.0,
        5.0
    ]

    lower, upper = (
        bootstrap_mean_confidence_interval(
            values,
            num_bootstrap_samples=1000,
            seed=42
        )
    )

    assert lower == pytest.approx(5.0)
    assert upper == pytest.approx(5.0)

    