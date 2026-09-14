import pandas as pd
import pytest

from src.trip_data_generator import (
    calculate_expected_travel_time,
    calculate_realized_travel_time,
    generate_trip_dataset
)


def test_expected_travel_time_formula():

    travel_time = (
        calculate_expected_travel_time(
            distance=30.0,
            traffic_level=3,
            road_type="residential",
            weather="rain"
        )
    )

    # Base:
    # 30 km / 30 km/h = 1 hour = 60 minutes
    #
    # Traffic factor:
    # 1 + 0.18 * (3 - 1) = 1.36
    #
    # Rain factor:
    # 1.15
    #
    # Expected:
    # 60 * 1.36 * 1.15 = 93.84

    assert travel_time == pytest.approx(
        93.84
    )


def test_realized_travel_time_adds_noise():

    expected = (
        calculate_expected_travel_time(
            distance=10.0,
            traffic_level=1,
            road_type="urban",
            weather="clear"
        )
    )

    realized = (
        calculate_realized_travel_time(
            distance=10.0,
            traffic_level=1,
            road_type="urban",
            weather="clear",
            noise=2.5
        )
    )

    assert realized == pytest.approx(
        expected + 2.5
    )


def test_realized_travel_time_has_minimum():

    realized = (
        calculate_realized_travel_time(
            distance=1.0,
            traffic_level=1,
            road_type="highway",
            weather="clear",
            noise=-100.0
        )
    )

    assert realized == pytest.approx(
        1.0
    )


def test_trip_dataset_is_reproducible():

    df_1 = generate_trip_dataset(
        num_trips=100,
        seed=42
    )

    df_2 = generate_trip_dataset(
        num_trips=100,
        seed=42
    )

    pd.testing.assert_frame_equal(
        df_1,
        df_2
    )
    