import random

import pandas as pd


# ==================================================
# Ground-truth travel-time model configuration
# ==================================================

ROAD_SPEEDS = {
    "residential": 30,
    "urban": 40,
    "arterial": 50,
    "highway": 85
}


WEATHER_FACTORS = {
    "clear": 1.00,
    "rain": 1.15,
    "heavy_rain": 1.30
}


RUSH_HOURS = {
    7,
    8,
    9,
    17,
    18,
    19
}


DEFAULT_NOISE_STD = 2.5


# ==================================================
# Traffic mechanics
# ==================================================

def calculate_traffic_factor(
    traffic_level: int
) -> float:
    return (
        1
        + 0.18
        * (traffic_level - 1)
    )


def sample_traffic_level(
    hour: int,
    weekday: int,
    rng: random.Random
) -> int:

    traffic_level = rng.randint(
        1,
        3
    )

    # Rush-hour congestion
    if hour in RUSH_HOURS:
        traffic_level += rng.randint(
            1,
            2
        )

    # Weekend traffic reduction
    if weekday >= 5:
        traffic_level -= 1

    # Keep traffic scale between 1 and 5
    traffic_level = max(
        1,
        min(
            traffic_level,
            5
        )
    )

    return traffic_level


# ==================================================
# Deterministic / expected travel time
# ==================================================

def calculate_expected_travel_time(
    distance: float,
    traffic_level: int,
    road_type: str,
    weather: str
) -> float:

    if distance < 0:
        raise ValueError(
            "Distance cannot be negative."
        )

    if traffic_level < 1 or traffic_level > 5:
        raise ValueError(
            "Traffic level must be between 1 and 5."
        )

    if road_type not in ROAD_SPEEDS:
        raise ValueError(
            f"Unknown road type: {road_type}"
        )

    if weather not in WEATHER_FACTORS:
        raise ValueError(
            f"Unknown weather condition: {weather}"
        )

    base_speed = (
        ROAD_SPEEDS[
            road_type
        ]
    )

    traffic_factor = (
        calculate_traffic_factor(
            traffic_level
        )
    )

    weather_factor = (
        WEATHER_FACTORS[
            weather
        ]
    )

    base_travel_time = (
        distance
        / base_speed
        * 60
    )

    expected_travel_time = (
        base_travel_time
        * traffic_factor
        * weather_factor
    )

    return expected_travel_time


# ==================================================
# Realized travel time
# ==================================================

def calculate_realized_travel_time(
    distance: float,
    traffic_level: int,
    road_type: str,
    weather: str,
    noise: float = 0.0
) -> float:

    expected_travel_time = (
        calculate_expected_travel_time(
            distance=distance,
            traffic_level=traffic_level,
            road_type=road_type,
            weather=weather
        )
    )

    realized_travel_time = (
        expected_travel_time
        + noise
    )

    return max(
        realized_travel_time,
        1.0
    )


# ==================================================
# Synthetic trip dataset generator
# ==================================================

def generate_trip_dataset(
    num_trips=10000,
    seed=42
):

    rng = random.Random(
        seed
    )

    trips = []

    road_types = list(
        ROAD_SPEEDS.keys()
    )

    weather_options = [
        "clear",
        "rain",
        "heavy_rain"
    ]

    weather_weights = [
        0.70,
        0.22,
        0.08
    ]

    for _ in range(
        num_trips
    ):

        # ------------------------------------------
        # Trip context
        # ------------------------------------------

        distance = rng.uniform(
            1,
            40
        )

        hour = rng.randint(
            0,
            23
        )

        weekday = rng.randint(
            0,
            6
        )

        road_type = rng.choice(
            road_types
        )

        weather = rng.choices(
            weather_options,
            weights=weather_weights
        )[0]

        traffic_level = (
            sample_traffic_level(
                hour=hour,
                weekday=weekday,
                rng=rng
            )
        )

        # ------------------------------------------
        # Irreducible stochastic component
        # ------------------------------------------

        noise = rng.gauss(
            0,
            DEFAULT_NOISE_STD
        )

        # ------------------------------------------
        # Ground-truth travel time
        # ------------------------------------------

        travel_time = (
            calculate_realized_travel_time(
                distance=distance,
                traffic_level=traffic_level,
                road_type=road_type,
                weather=weather,
                noise=noise
            )
        )

        trips.append(
            {
                "distance":
                    distance,

                "hour":
                    hour,

                "weekday":
                    weekday,

                "traffic_level":
                    traffic_level,

                "road_type":
                    road_type,

                "weather":
                    weather,

                "travel_time":
                    travel_time
            }
        )

    return pd.DataFrame(
        trips
    )

