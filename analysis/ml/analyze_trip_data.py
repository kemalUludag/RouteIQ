import pandas as pd


trips_df = pd.read_csv(
    "data/trip_data.csv"
)


# -------------------------
# Basic information
# -------------------------

print("=== DATASET INFO ===")
print(f"Rows: {len(trips_df)}")
print(f"Columns: {len(trips_df.columns)}")

print()


# -------------------------
# Numerical correlations
# -------------------------

numerical_columns = [
    "distance",
    "hour",
    "weekday",
    "traffic_level",
    "travel_time"
]

correlations = (
    trips_df[numerical_columns]
    .corr()["travel_time"]
    .sort_values(ascending=False)
)

print("=== CORRELATION WITH TRAVEL TIME ===")
print(correlations)

print()


# -------------------------
# Road type analysis
# -------------------------

road_type_summary = (
    trips_df
    .groupby("road_type")["travel_time"]
    .agg(["count", "mean", "median"])
    .sort_values("mean", ascending=False)
)

print("=== TRAVEL TIME BY ROAD TYPE ===")
print(road_type_summary)

print()


# -------------------------
# Weather analysis
# -------------------------

weather_summary = (
    trips_df
    .groupby("weather")["travel_time"]
    .agg(["count", "mean", "median"])
    .sort_values("mean", ascending=False)
)

print("=== TRAVEL TIME BY WEATHER ===")
print(weather_summary)

print()


# -------------------------
# Traffic analysis
# -------------------------

traffic_summary = (
    trips_df
    .groupby("traffic_level")["travel_time"]
    .agg(["count", "mean", "median"])
)

print("=== TRAVEL TIME BY TRAFFIC LEVEL ===")
print(traffic_summary)

print()


# -------------------------
# Rush-hour feature
# -------------------------

rush_hours = [7, 8, 9, 17, 18, 19]

trips_df["is_rush_hour"] = (
    trips_df["hour"]
    .isin(rush_hours)
    .astype(int)
)

rush_hour_summary = (
    trips_df
    .groupby("is_rush_hour")["travel_time"]
    .agg(["count", "mean", "median"])
)

print("=== RUSH HOUR ANALYSIS ===")
print(rush_hour_summary)


