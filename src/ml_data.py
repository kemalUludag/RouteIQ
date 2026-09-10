from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "travel_time"


@dataclass
class MLDataSplit:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame

    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def load_trip_data(
    filepath: str = "data/trip_data.csv"
) -> pd.DataFrame:
    return pd.read_csv(filepath)


def split_trip_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    validation_size: float = 0.20,
    random_state: int = 42
) -> MLDataSplit:

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[
        TARGET_COLUMN
    ]

    X_train_validation, X_test, y_train_validation, y_test = (
        train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state
        )
    )

    validation_fraction = (
        validation_size
        /
        (1.0 - test_size)
    )

    X_train, X_validation, y_train, y_validation = (
        train_test_split(
            X_train_validation,
            y_train_validation,
            test_size=validation_fraction,
            random_state=random_state
        )
    )

    return MLDataSplit(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test
    )

