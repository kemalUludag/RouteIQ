import pandas as pd

from src.ml_data import (
    TARGET_COLUMN,
    split_trip_data
)


def create_test_dataframe(
    num_rows=100
):
    return pd.DataFrame(
        {
            "distance": [
                float(i)
                for i in range(num_rows)
            ],

            "hour": [
                i % 24
                for i in range(num_rows)
            ],

            "weekday": [
                i % 7
                for i in range(num_rows)
            ],

            "traffic": [
                (i % 5) + 1
                for i in range(num_rows)
            ],

            "road_type": [
                "urban"
                if i % 2 == 0
                else "highway"
                for i in range(num_rows)
            ],

            "weather": [
                "clear"
                if i % 3 == 0
                else "rain"
                for i in range(num_rows)
            ],

            "travel_time": [
                float(i + 10)
                for i in range(num_rows)
            ]
        }
    )


def test_split_trip_data_has_correct_sizes():
    df = create_test_dataframe(
        num_rows=100
    )

    split = split_trip_data(
        df,
        test_size=0.20,
        validation_size=0.20,
        random_state=42
    )

    assert len(split.X_train) == 60
    assert len(split.X_validation) == 20
    assert len(split.X_test) == 20

    assert len(split.y_train) == 60
    assert len(split.y_validation) == 20
    assert len(split.y_test) == 20


def test_target_is_removed_from_features():
    df = create_test_dataframe()

    split = split_trip_data(
        df
    )

    assert (
        TARGET_COLUMN
        not in split.X_train.columns
    )

    assert (
        TARGET_COLUMN
        not in split.X_validation.columns
    )

    assert (
        TARGET_COLUMN
        not in split.X_test.columns
    )


def test_split_is_reproducible():
    df = create_test_dataframe()

    split_1 = split_trip_data(
        df,
        random_state=42
    )

    split_2 = split_trip_data(
        df,
        random_state=42
    )

    assert (
        split_1.X_train.index.tolist()
        ==
        split_2.X_train.index.tolist()
    )

    assert (
        split_1.X_validation.index.tolist()
        ==
        split_2.X_validation.index.tolist()
    )

    assert (
        split_1.X_test.index.tolist()
        ==
        split_2.X_test.index.tolist()
    )
    