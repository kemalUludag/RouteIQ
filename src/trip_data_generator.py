import random
import pandas as pd


def generate_trip_dataset(
    num_trips=10000,
    seed=42
):
    random.seed(seed)

    road_speeds = {
        "residential": 30,
        "urban": 40,
        "arterial": 50,
        "highway": 85
    }

    weather_factors = {
        "clear": 1.00,
        "rain": 1.15,
        "heavy_rain": 1.30
    }

    trips = []

    for _ in range(num_trips):
        distance = random.uniform(1, 40)

        hour = random.randint(0, 23)

        weekday = random.randint(0, 6)

        road_type = random.choice(
            list(road_speeds.keys())
        )

        weather = random.choices(
            ["clear", "rain", "heavy_rain"],
            weights=[0.70, 0.22, 0.08]
        )[0]

        traffic_level = random.randint(1, 3)

        if hour in [7, 8, 9, 17, 18, 19]:
            traffic_level += random.randint(1, 2)

        if weekday >= 5:
            traffic_level -= 1

        traffic_level = max(
            1,
            min(traffic_level, 5)
        )

        base_speed = road_speeds[road_type]

        traffic_factor = (
            1 + 0.18 * (traffic_level - 1)
        )

        weather_factor = weather_factors[weather]

        base_travel_time = (
            distance / base_speed
        ) * 60

        travel_time = (
            base_travel_time
            * traffic_factor
            * weather_factor
        )

        noise = random.gauss(
            0,
            2.5
        )

        travel_time += noise

        travel_time = max(
            travel_time,
            1
        )

        trips.append({
            "distance": distance,
            "hour": hour,
            "weekday": weekday,
            "traffic_level": traffic_level,
            "road_type": road_type,
            "weather": weather,
            "travel_time": travel_time
        })

    return pd.DataFrame(trips)
