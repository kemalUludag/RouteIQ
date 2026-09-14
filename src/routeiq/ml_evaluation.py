from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)


@dataclass
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float


def evaluate_regression(
    y_true,
    y_pred
) -> RegressionMetrics:
    return RegressionMetrics(
        mae=float(
            mean_absolute_error(
                y_true,
                y_pred
            )
        ),
        rmse=float(
            root_mean_squared_error(
                y_true,
                y_pred
            )
        ),
        r2=float(
            r2_score(
                y_true,
                y_pred
            )
        )
    )


def calculate_prediction_errors(
    y_true,
    y_pred
):
    y_true_array = np.asarray(
        y_true
    )

    y_pred_array = np.asarray(
        y_pred
    )

    return (
        y_pred_array
        - y_true_array
    )

