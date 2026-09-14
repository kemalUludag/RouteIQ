import numpy as np
import pytest

from src.travel_time_matrix import (
    build_free_flow_travel_time_matrix
)

from src.models import (
    CVRPInstance
)

from src.travel_time_matrix import (
    RoutingContextScenario,
    generate_routing_context,
    build_edge_feature_frame,
    build_true_expected_travel_time_matrix,
    build_predicted_travel_time_matrix,
    build_realized_travel_time_matrix
)


def create_small_instance():

    return CVRPInstance(
        name="travel_time_matrix_test",
        points=[
            (0, 0),
            (3, 4),
            (6, 8)
        ],
        demands=[
            0,
            2,
            3
        ],
        num_vehicles=1,
        vehicle_capacities=[
            10
        ]
    )


def test_routing_context_is_reproducible():

    instance = (
        create_small_instance()
    )

    scenario_1 = (
        generate_routing_context(
            instance=instance,
            hour=18,
            weekday=2,
            weather="rain",
            seed=42
        )
    )

    scenario_2 = (
        generate_routing_context(
            instance=instance,
            hour=18,
            weekday=2,
            weather="rain",
            seed=42
        )
    )

    assert (
        scenario_1.road_type_matrix
        ==
        scenario_2.road_type_matrix
    )

    assert (
        scenario_1.traffic_level_matrix
        ==
        scenario_2.traffic_level_matrix
    )


def test_routing_context_is_symmetric():

    instance = (
        create_small_instance()
    )

    scenario = (
        generate_routing_context(
            instance=instance,
            hour=18,
            weekday=2,
            weather="rain",
            seed=42
        )
    )

    for i in range(
        instance.num_nodes
    ):
        for j in range(
            instance.num_nodes
        ):

            assert (
                scenario.road_type_matrix[i][j]
                ==
                scenario.road_type_matrix[j][i]
            )

            assert (
                scenario.traffic_level_matrix[i][j]
                ==
                scenario.traffic_level_matrix[j][i]
            )


def test_edge_feature_frame_has_all_directed_edges():

    instance = (
        create_small_instance()
    )

    scenario = (
        generate_routing_context(
            instance=instance,
            hour=18,
            weekday=2,
            weather="rain",
            seed=42
        )
    )

    feature_frame = (
        build_edge_feature_frame(
            instance,
            scenario
        )
    )

    # 3 nodes:
    #
    # 3 * (3 - 1)
    # = 6 directed edges

    assert len(
        feature_frame
    ) == 6

    assert set(
        feature_frame.columns
    ) == {
        "origin",
        "destination",
        "distance",
        "hour",
        "weekday",
        "traffic_level",
        "road_type",
        "weather"
    }


def test_true_expected_matrix_has_zero_diagonal():

    instance = (
        create_small_instance()
    )

    scenario = (
        generate_routing_context(
            instance=instance,
            hour=18,
            weekday=2,
            weather="rain",
            seed=42
        )
    )

    matrix = (
        build_true_expected_travel_time_matrix(
            instance,
            scenario
        )
    )

    for i in range(
        instance.num_nodes
    ):
        assert matrix[i][i] == pytest.approx(
            0.0
        )


def test_true_expected_matrix_formula():

    instance = (
        create_small_instance()
    )

    scenario = RoutingContextScenario(
        hour=12,
        weekday=1,
        weather="clear",

        road_type_matrix=[
            [
                None,
                "urban",
                "urban"
            ],
            [
                "urban",
                None,
                "urban"
            ],
            [
                "urban",
                "urban",
                None
            ]
        ],

        traffic_level_matrix=[
            [
                0,
                1,
                1
            ],
            [
                1,
                0,
                1
            ],
            [
                1,
                1,
                0
            ]
        ]
    )

    matrix = (
        build_true_expected_travel_time_matrix(
            instance,
            scenario
        )
    )

    # Distance from node 0 to node 1 = 5 km
    #
    # Urban speed = 40 km/h
    #
    # Traffic factor = 1
    # Weather factor = 1
    #
    # 5 / 40 * 60 = 7.5 minutes

    assert matrix[0][1] == pytest.approx(
        7.5
    )

    assert matrix[1][0] == pytest.approx(
        7.5
    )


def test_predicted_matrix_maps_predictions():

    instance = (
        create_small_instance()
    )

    scenario = (
        generate_routing_context(
            instance=instance,
            hour=18,
            weekday=2,
            weather="rain",
            seed=42
        )
    )

    class ConstantPredictionModel:

        def predict(
            self,
            X
        ):
            return np.full(
                len(X),
                12.5
            )

    model = (
        ConstantPredictionModel()
    )

    matrix = (
        build_predicted_travel_time_matrix(
            instance=instance,
            scenario=scenario,
            model=model
        )
    )

    for i in range(
        instance.num_nodes
    ):
        for j in range(
            instance.num_nodes
        ):

            if i == j:
                assert matrix[i][j] == pytest.approx(
                    0.0
                )

            else:
                assert matrix[i][j] == pytest.approx(
                    12.5
                )

def test_same_road_seed_preserves_network_structure():

    instance = (
        create_small_instance()
    )

    scenario_1 = (
        generate_routing_context(
            instance=instance,
            hour=12,
            weekday=1,
            weather="clear",
            road_seed=100,
            traffic_seed=200
        )
    )

    scenario_2 = (
        generate_routing_context(
            instance=instance,
            hour=18,
            weekday=1,
            weather="rain",
            road_seed=100,
            traffic_seed=300
        )
    )

    assert (
        scenario_1.road_type_matrix
        ==
        scenario_2.road_type_matrix
    )


def test_different_traffic_seed_can_change_traffic():

    instance = (
        create_small_instance()
    )

    scenario_1 = (
        generate_routing_context(
            instance=instance,
            hour=18,
            weekday=2,
            weather="rain",
            road_seed=100,
            traffic_seed=200
        )
    )

    scenario_2 = (
        generate_routing_context(
            instance=instance,
            hour=18,
            weekday=2,
            weather="rain",
            road_seed=100,
            traffic_seed=201
        )
    )

    assert (
        scenario_1.road_type_matrix
        ==
        scenario_2.road_type_matrix
    )

    assert (
        scenario_1.traffic_level_matrix
        !=
        scenario_2.traffic_level_matrix
    )

    
def test_free_flow_matrix_uses_road_speed():

    instance = (
        create_small_instance()
    )

    scenario = RoutingContextScenario(
        hour=18,
        weekday=2,
        weather="heavy_rain",

        road_type_matrix=[
            [
                None,
                "urban",
                "highway"
            ],
            [
                "urban",
                None,
                "residential"
            ],
            [
                "highway",
                "residential",
                None
            ]
        ],

        traffic_level_matrix=[
            [
                0,
                5,
                5
            ],
            [
                5,
                0,
                5
            ],
            [
                5,
                5,
                0
            ]
        ]
    )

    matrix = (
        build_free_flow_travel_time_matrix(
            instance,
            scenario
        )
    )

    # Node 0 -> node 1:
    #
    # coordinates:
    # (0, 0) -> (3, 4)
    #
    # distance = 5
    #
    # urban speed = 40 km/h
    #
    # free-flow time:
    #
    # 5 / 40 * 60 = 7.5 minutes
    #
    # Traffic = 5 and heavy rain must NOT
    # affect the free-flow baseline.

    assert matrix[0][1] == pytest.approx(
        7.5
    )

    assert matrix[1][0] == pytest.approx(
        7.5
    )


def test_realized_matrix_is_reproducible():

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

    matrix_1 = (
        build_realized_travel_time_matrix(
            instance=instance,
            scenario=scenario,
            seed=999
        )
    )

    matrix_2 = (
        build_realized_travel_time_matrix(
            instance=instance,
            scenario=scenario,
            seed=999
        )
    )

    assert (
        matrix_1
        ==
        matrix_2
    )


def test_realized_matrix_is_symmetric():

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

    matrix = (
        build_realized_travel_time_matrix(
            instance=instance,
            scenario=scenario,
            seed=999
        )
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
                == pytest.approx(
                    matrix[j][i]
                )
            )


def test_zero_noise_matches_expected_matrix():

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

    expected_matrix = (
        build_true_expected_travel_time_matrix(
            instance=instance,
            scenario=scenario
        )
    )

    realized_matrix = (
        build_realized_travel_time_matrix(
            instance=instance,
            scenario=scenario,
            seed=999,
            noise_std=0.0
        )
    )

    for i in range(
        instance.num_nodes
    ):

        for j in range(
            instance.num_nodes
        ):

            assert (
                realized_matrix[i][j]
                == pytest.approx(
                    expected_matrix[i][j]
                )
            )

            