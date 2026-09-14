from dataclasses import dataclass
import random

from .distance import (
    create_distance_matrix
)

from .models import (
    CVRPInstance
)

from .travel_time_matrix import (
    RoutingContextScenario
)

from .trip_data_generator import (
    calculate_expected_travel_time
)


# ==================================================
# Stress-test configuration
# ==================================================

@dataclass(frozen=True)
class StressTestConfig:

    # Minimum edge-level uncertainty
    # measured in minutes.
    base_noise_std_minutes: float

    # Additional uncertainty proportional
    # to expected travel time.
    #
    # Example:
    #
    # expected = 20 minutes
    # relative_noise_scale = 0.08
    #
    # contribution = 1.6 minutes
    relative_noise_scale: float

    # Standard deviation of the latent
    # network-wide congestion shock.
    #
    # Example:
    #
    # 0.08 roughly represents an 8%
    # common multiplicative disturbance
    # before edge exposure adjustment.
    common_shock_std: float

    # High-traffic edges are more exposed
    # to the common congestion shock.
    traffic_sensitivity_scale: float

    # Defensive clipping prevents extreme
    # Gaussian shock draws from creating
    # unrealistic negative / explosive factors.
    min_shock_factor: float = 0.50
    max_shock_factor: float = 1.75

    def __post_init__(self):

        if (
            self.base_noise_std_minutes
            < 0
        ):
            raise ValueError(
                "base_noise_std_minutes "
                "cannot be negative."
            )

        if (
            self.relative_noise_scale
            < 0
        ):
            raise ValueError(
                "relative_noise_scale "
                "cannot be negative."
            )

        if (
            self.common_shock_std
            < 0
        ):
            raise ValueError(
                "common_shock_std "
                "cannot be negative."
            )

        if (
            self.traffic_sensitivity_scale
            < 0
        ):
            raise ValueError(
                "traffic_sensitivity_scale "
                "cannot be negative."
            )

        if (
            self.min_shock_factor
            <= 0
        ):
            raise ValueError(
                "min_shock_factor "
                "must be positive."
            )

        if (
            self.max_shock_factor
            < self.min_shock_factor
        ):
            raise ValueError(
                "max_shock_factor must be "
                "greater than or equal to "
                "min_shock_factor."
            )


# ==================================================
# Stress-test presets
#
# IMPORTANT:
#
# These are deliberately designed synthetic
# stress intensities.
#
# They are NOT claimed to be calibrated
# from real traffic data.
# ==================================================

MILD_STRESS = StressTestConfig(
    base_noise_std_minutes=0.75,
    relative_noise_scale=0.04,
    common_shock_std=0.04,
    traffic_sensitivity_scale=0.10
)


MODERATE_STRESS = StressTestConfig(
    base_noise_std_minutes=1.00,
    relative_noise_scale=0.08,
    common_shock_std=0.08,
    traffic_sensitivity_scale=0.15
)


SEVERE_STRESS = StressTestConfig(
    base_noise_std_minutes=1.50,
    relative_noise_scale=0.12,
    common_shock_std=0.12,
    traffic_sensitivity_scale=0.20
)


STRESS_CONFIGS = {
    "mild":
        MILD_STRESS,

    "moderate":
        MODERATE_STRESS,

    "severe":
        SEVERE_STRESS
}


# ==================================================
# One complete realization
# ==================================================

@dataclass
class StressRealization:
    travel_time_matrix: list[list[float]]
    common_shock: float


# ==================================================
# Heteroscedastic edge uncertainty
# ==================================================

def calculate_edge_noise_std(
    expected_travel_time: float,
    config: StressTestConfig
) -> float:

    if expected_travel_time < 0:
        raise ValueError(
            "Expected travel time "
            "cannot be negative."
        )

    return (
        config.base_noise_std_minutes
        +
        config.relative_noise_scale
        * expected_travel_time
    )


# ==================================================
# Exposure to common congestion shock
# ==================================================

def calculate_common_shock_factor(
    common_shock: float,
    traffic_level: int,
    config: StressTestConfig
) -> float:

    if (
        traffic_level < 1
        or traffic_level > 5
    ):
        raise ValueError(
            "Traffic level must be "
            "between 1 and 5."
        )

    # ==========================================
    # Higher current congestion makes an edge
    # more sensitive to a city/network-wide
    # congestion disturbance.
    #
    # traffic 1:
    # sensitivity = 1.00
    #
    # traffic 5, scale=.15:
    # sensitivity = 1.60
    # ==========================================

    sensitivity = (
        1.0
        +
        config.traffic_sensitivity_scale
        * (
            traffic_level
            - 1
        )
    )

    shock_factor = (
        1.0
        +
        common_shock
        * sensitivity
    )

    # ==========================================
    # Clip extreme Gaussian draws
    # ==========================================

    shock_factor = max(
        config.min_shock_factor,
        min(
            shock_factor,
            config.max_shock_factor
        )
    )

    return shock_factor


# ==================================================
# Build one stress realization
# ==================================================

def build_stress_realization(
    instance: CVRPInstance,
    scenario: RoutingContextScenario,
    seed: int,
    config: StressTestConfig
) -> StressRealization:

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
    # ONE latent shock is sampled for the
    # entire network realization.
    #
    # This introduces correlation across edges.
    # ==========================================

    common_shock = rng.gauss(
        0.0,
        config.common_shock_std
    )

    # ==========================================
    # Symmetric network remains intentional
    # at the current RouteIQ stage.
    # ==========================================

    for i in range(num_nodes):

        for j in range(
            i + 1,
            num_nodes
        ):

            traffic_level = (
                scenario
                .traffic_level_matrix[i][j]
            )

            road_type = (
                scenario
                .road_type_matrix[i][j]
            )

            # ==================================
            # Original expected travel time
            #
            # This remains the center of the
            # shifted realization distribution.
            # ==================================

            expected_travel_time = (
                calculate_expected_travel_time(
                    distance=(
                        distance_matrix[i][j]
                    ),

                    traffic_level=(
                        traffic_level
                    ),

                    road_type=(
                        road_type
                    ),

                    weather=(
                        scenario.weather
                    )
                )
            )

            # ==================================
            # Correlated network shock
            # ==================================

            shock_factor = (
                calculate_common_shock_factor(
                    common_shock=(
                        common_shock
                    ),

                    traffic_level=(
                        traffic_level
                    ),

                    config=(
                        config
                    )
                )
            )

            shocked_expected_time = (
                expected_travel_time
                * shock_factor
            )

            # ==================================
            # Heteroscedastic edge residual
            #
            # Longer trips now receive
            # larger absolute uncertainty.
            # ==================================

            edge_noise_std = (
                calculate_edge_noise_std(
                    expected_travel_time=(
                        expected_travel_time
                    ),

                    config=(
                        config
                    )
                )
            )

            edge_noise = rng.gauss(
                0.0,
                edge_noise_std
            )

            realized_travel_time = (
                shocked_expected_time
                +
                edge_noise
            )

            realized_travel_time = max(
                realized_travel_time,
                1.0
            )

            # ==================================
            # Preserve symmetry
            # ==================================

            realized_matrix[i][j] = (
                realized_travel_time
            )

            realized_matrix[j][i] = (
                realized_travel_time
            )

    return StressRealization(
        travel_time_matrix=(
            realized_matrix
        ),
        common_shock=(
            common_shock
        )
    )

