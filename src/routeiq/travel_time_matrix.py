from dataclasses import dataclass
import random

import pandas as pd

from .distance import (
    create_distance_matrix
)

from .models import (
    CVRPInstance
)

from .trip_data_generator import (
    ROAD_SPEEDS,
    DEFAULT_NOISE_STD,
    calculate_expected_travel_time,
    calculate_realized_travel_time,
    sample_traffic_level
)



# ==================================================
# Routing scenario representation
# ==================================================

@dataclass
class RoutingContextScenario:
    hour: int
    weekday: int
    weather: str

    road_type_matrix: list[list[str | None]]
    traffic_level_matrix: list[list[int]]


# ==================================================
# Persistent network structure
# ==================================================

def generate_road_type_matrix(
    instance: CVRPInstance,
    seed: int = 42
) -> list[list[str | None]]:

    rng = random.Random(
        seed
    )

    num_nodes = (
        instance.num_nodes
    )

    road_types = list(
        ROAD_SPEEDS.keys()
    )

    road_type_matrix = [
        [
            None
            for _ in range(num_nodes)
        ]
        for _ in range(num_nodes)
    ]

    # ==========================================
    # Symmetric road structure
    # ==========================================

    for i in range(num_nodes):
        for j in range(
            i + 1,
            num_nodes
        ):

            road_type = (
                rng.choice(
                    road_types
                )
            )

            road_type_matrix[i][j] = (
                road_type
            )

            road_type_matrix[j][i] = (
                road_type
            )

    return road_type_matrix


# ==================================================
# Dynamic traffic state
# ==================================================

def generate_traffic_level_matrix(
    instance: CVRPInstance,
    hour: int,
    weekday: int,
    seed: int = 42
) -> list[list[int]]:

    rng = random.Random(
        seed
    )

    num_nodes = (
        instance.num_nodes
    )

    traffic_level_matrix = [
        [
            0
            for _ in range(num_nodes)
        ]
        for _ in range(num_nodes)
    ]

    # ==========================================
    # Symmetric traffic for current project stage
    # ==========================================

    for i in range(num_nodes):
        for j in range(
            i + 1,
            num_nodes
        ):

            traffic_level = (
                sample_traffic_level(
                    hour=hour,
                    weekday=weekday,
                    rng=rng
                )
            )

            traffic_level_matrix[i][j] = (
                traffic_level
            )

            traffic_level_matrix[j][i] = (
                traffic_level
            )

    return traffic_level_matrix


# ==================================================
# Complete routing context
# ==================================================

def generate_routing_context(
    instance: CVRPInstance,
    hour: int,
    weekday: int,
    weather: str,
    seed: int = 42,
    road_seed: int | None = None,
    traffic_seed: int | None = None
) -> RoutingContextScenario:

    # ==========================================
    # Backward-compatible defaults
    # ==========================================

    if road_seed is None:
        road_seed = seed

    if traffic_seed is None:
        traffic_seed = seed

    # ==========================================
    # Persistent network characteristics
    # ==========================================

    road_type_matrix = (
        generate_road_type_matrix(
            instance=instance,
            seed=road_seed
        )
    )

    # ==========================================
    # Dynamic traffic characteristics
    # ==========================================

    traffic_level_matrix = (
        generate_traffic_level_matrix(
            instance=instance,
            hour=hour,
            weekday=weekday,
            seed=traffic_seed
        )
    )

    return RoutingContextScenario(
        hour=hour,
        weekday=weekday,
        weather=weather,
        road_type_matrix=road_type_matrix,
        traffic_level_matrix=traffic_level_matrix
    )


# ==================================================
# Edge-level feature table
# ==================================================

def build_edge_feature_frame(
    instance: CVRPInstance,
    scenario: RoutingContextScenario
) -> pd.DataFrame:

    distance_matrix = (
        create_distance_matrix(
            instance.points
        )
    )

    rows = []

    for i in range(
        instance.num_nodes
    ):
        for j in range(
            instance.num_nodes
        ):

            if i == j:
                continue

            rows.append(
                {
                    "origin":
                        i,

                    "destination":
                        j,

                    "distance":
                        distance_matrix[i][j],

                    "hour":
                        scenario.hour,

                    "weekday":
                        scenario.weekday,

                    "traffic_level":
                        scenario
                        .traffic_level_matrix[i][j],

                    "road_type":
                        scenario
                        .road_type_matrix[i][j],

                    "weather":
                        scenario.weather
                }
            )

    return pd.DataFrame(
        rows
    )
def build_free_flow_travel_time_matrix(
    instance: CVRPInstance,
    scenario: RoutingContextScenario
) -> list[list[float]]:

    distance_matrix = (
        create_distance_matrix(
            instance.points
        )
    )

    num_nodes = (
        instance.num_nodes
    )

    travel_time_matrix = [
        [
            0.0
            for _ in range(num_nodes)
        ]
        for _ in range(num_nodes)
    ]

    for i in range(num_nodes):
        for j in range(num_nodes):

            if i == j:
                continue

            road_type = (
                scenario
                .road_type_matrix[i][j]
            )

            base_speed = (
                ROAD_SPEEDS[
                    road_type
                ]
            )

            travel_time = (
                distance_matrix[i][j]
                / base_speed
                * 60
            )

            travel_time_matrix[i][j] = (
                travel_time
            )

    return travel_time_matrix


# ==================================================
# Ground-truth expected travel-time matrix
# ==================================================

def build_true_expected_travel_time_matrix(
    instance: CVRPInstance,
    scenario: RoutingContextScenario
) -> list[list[float]]:

    distance_matrix = (
        create_distance_matrix(
            instance.points
        )
    )

    num_nodes = (
        instance.num_nodes
    )

    travel_time_matrix = [
        [
            0.0
            for _ in range(num_nodes)
        ]
        for _ in range(num_nodes)
    ]

    for i in range(num_nodes):
        for j in range(num_nodes):

            if i == j:
                continue

            travel_time = (
                calculate_expected_travel_time(
                    distance=(
                        distance_matrix[i][j]
                    ),

                    traffic_level=(
                        scenario
                        .traffic_level_matrix[i][j]
                    ),

                    road_type=(
                        scenario
                        .road_type_matrix[i][j]
                    ),

                    weather=(
                        scenario.weather
                    )
                )
            )

            travel_time_matrix[i][j] = (
                travel_time
            )

    return travel_time_matrix


# ==================================================
# ML-predicted travel-time matrix
# ==================================================

def build_predicted_travel_time_matrix(
    instance: CVRPInstance,
    scenario: RoutingContextScenario,
    model
) -> list[list[float]]:

    feature_frame = (
        build_edge_feature_frame(
            instance,
            scenario
        )
    )

    # ==========================================
    # origin / destination are identifiers only.
    #
    # They are NOT predictive features.
    # ==========================================

    model_features = (
        feature_frame[
            [
                "distance",
                "hour",
                "weekday",
                "traffic_level",
                "road_type",
                "weather"
            ]
        ]
    )

    predictions = (
        model.predict(
            model_features
        )
    )

    num_nodes = (
        instance.num_nodes
    )

    travel_time_matrix = [
        [
            0.0
            for _ in range(num_nodes)
        ]
        for _ in range(num_nodes)
    ]

    for row, prediction in zip(
        feature_frame.itertuples(
            index=False
        ),
        predictions
    ):

        travel_time_matrix[
            row.origin
        ][
            row.destination
        ] = max(
            float(prediction),
            1.0
        )

    return travel_time_matrix

def build_realized_travel_time_matrix(
    instance: CVRPInstance,
    scenario: RoutingContextScenario,
    seed: int,
    noise_std: float = DEFAULT_NOISE_STD
) -> list[list[float]]:

    rng = random.Random(
        seed
    )

    distance_matrix = (
        create_distance_matrix(
            instance.points
        )
    )

    num_nodes = (
        instance.num_nodes
    )

    realized_matrix = [
        [
            0.0
            for _ in range(num_nodes)
        ]
        for _ in range(num_nodes)
    ]

    # ==========================================
    # Current project stage:
    #
    # The routing network is symmetric.
    #
    # Therefore one random realization is
    # sampled for edge {i, j} and reused for
    # both directions.
    #
    # Later we can explicitly introduce
    # directional / asymmetric uncertainty.
    # ==========================================

    for i in range(num_nodes):

        for j in range(
            i + 1,
            num_nodes
        ):

            noise = rng.gauss(
                0.0,
                noise_std
            )

            realized_travel_time = (
                calculate_realized_travel_time(
                    distance=(
                        distance_matrix[i][j]
                    ),

                    traffic_level=(
                        scenario
                        .traffic_level_matrix[i][j]
                    ),

                    road_type=(
                        scenario
                        .road_type_matrix[i][j]
                    ),

                    weather=(
                        scenario.weather
                    ),

                    noise=(
                        noise
                    )
                )
            )

            realized_matrix[i][j] = (
                realized_travel_time
            )

            realized_matrix[j][i] = (
                realized_travel_time
            )

    return realized_matrix

