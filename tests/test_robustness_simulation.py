import pytest

from src.models import (
    CVRPInstance
)

from src.robustness_simulation import (
    StressTestConfig,
    MODERATE_STRESS,
    calculate_edge_noise_std,
    calculate_common_shock_factor,
    build_stress_realization
)

from src.travel_time_matrix import (
    generate_routing_context,
    build_true_expected_travel_time_matrix
)


def create_small_instance():

    return CVRPInstance(
        name="robustness_test",

        points=[
            (0, 0),
            (3, 4),
            (6, 8)
        ],

        demands=[
            0,
            1,
            1
        ],

        num_vehicles=1,

        vehicle_capacities=[
            10
        ]
    )


def test_longer_expected_time_has_more_uncertainty():

    short_trip_std = (
        calculate_edge_noise_std(
            expected_travel_time=5.0,
            config=MODERATE_STRESS
        )
    )

    long_trip_std = (
        calculate_edge_noise_std(
            expected_travel_time=30.0,
            config=MODERATE_STRESS
        )
    )

    assert (
        long_trip_std
        >
        short_trip_std
    )


def test_high_traffic_is_more_exposed_to_positive_common_shock():

    low_traffic_factor = (
        calculate_common_shock_factor(
            common_shock=0.10,
            traffic_level=1,
            config=MODERATE_STRESS
        )
    )

    high_traffic_factor = (
        calculate_common_shock_factor(
            common_shock=0.10,
            traffic_level=5,
            config=MODERATE_STRESS
        )
    )

    assert (
        high_traffic_factor
        >
        low_traffic_factor
    )


def test_stress_realization_is_reproducible():

    instance = (
        create_small_instance()
    )

    scenario = (
        generate_routing_context(
            instance=instance,
            hour=18,
            weekday=2,
            weather="clear",
            road_seed=100,
            traffic_seed=200
        )
    )

    realization_1 = (
        build_stress_realization(
            instance=instance,
            scenario=scenario,
            seed=999,
            config=MODERATE_STRESS
        )
    )

    realization_2 = (
        build_stress_realization(
            instance=instance,
            scenario=scenario,
            seed=999,
            config=MODERATE_STRESS
        )
    )

    assert (
        realization_1.common_shock
        ==
        pytest.approx(
            realization_2.common_shock
        )
    )

    assert (
        realization_1.travel_time_matrix
        ==
        realization_2.travel_time_matrix
    )


def test_stress_realization_is_symmetric():

    instance = (
        create_small_instance()
    )

    scenario = (
        generate_routing_context(
            instance=instance,
            hour=18,
            weekday=2,
            weather="clear",
            road_seed=100,
            traffic_seed=200
        )
    )

    realization = (
        build_stress_realization(
            instance=instance,
            scenario=scenario,
            seed=999,
            config=MODERATE_STRESS
        )
    )

    matrix = (
        realization.travel_time_matrix
    )

    for i in range(
        instance.num_nodes
    ):

        assert (
            matrix[i][i]
            == pytest.approx(
                0.0
            )
        )

        for j in range(
            instance.num_nodes
        ):

            assert (
                matrix[i][j]
                ==
                pytest.approx(
                    matrix[j][i]
                )
            )


def test_zero_stress_matches_expected_matrix():

    instance = (
        create_small_instance()
    )

    scenario = (
        generate_routing_context(
            instance=instance,
            hour=18,
            weekday=2,
            weather="clear",
            road_seed=100,
            traffic_seed=200
        )
    )

    zero_stress = StressTestConfig(
        base_noise_std_minutes=0.0,
        relative_noise_scale=0.0,
        common_shock_std=0.0,
        traffic_sensitivity_scale=0.0
    )

    expected_matrix = (
        build_true_expected_travel_time_matrix(
            instance=instance,
            scenario=scenario
        )
    )

    realization = (
        build_stress_realization(
            instance=instance,
            scenario=scenario,
            seed=999,
            config=zero_stress
        )
    )

    realized_matrix = (
        realization.travel_time_matrix
    )

    for i in range(
        instance.num_nodes
    ):

        for j in range(
            instance.num_nodes
        ):

            assert (
                realized_matrix[i][j]
                ==
                pytest.approx(
                    expected_matrix[i][j]
                )
            )



            