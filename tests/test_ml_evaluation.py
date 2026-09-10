import pytest

from src.ml_evaluation import (
    evaluate_regression,
    calculate_prediction_errors
)


def test_evaluate_regression_perfect_predictions():
    y_true = [
        10.0,
        20.0,
        30.0
    ]

    y_pred = [
        10.0,
        20.0,
        30.0
    ]

    metrics = evaluate_regression(
        y_true,
        y_pred
    )

    assert metrics.mae == pytest.approx(
        0.0
    )

    assert metrics.rmse == pytest.approx(
        0.0
    )

    assert metrics.r2 == pytest.approx(
        1.0
    )


def test_calculate_prediction_errors():
    y_true = [
        10.0,
        20.0,
        30.0
    ]

    y_pred = [
        12.0,
        18.0,
        35.0
    ]

    errors = calculate_prediction_errors(
        y_true,
        y_pred
    )

    assert errors.tolist() == [
        2.0,
        -2.0,
        5.0
    ]

    