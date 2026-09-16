from pathlib import Path

from routeiq.trip_data_generator import generate_trip_dataset


OUTPUT_PATH = Path("data/trip_data.csv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


trips_df = generate_trip_dataset(
    num_trips=10000,
    seed=42
)


print("=== DATASET SHAPE ===")
print(trips_df.shape)

print()

print("=== FIRST 5 ROWS ===")
print(trips_df.head())

print()

print("=== DATA TYPES ===")
print(trips_df.dtypes)

print()

print("=== MISSING VALUES ===")
print(trips_df.isnull().sum())

print()

print("=== NUMERICAL SUMMARY ===")
print(trips_df.describe())


trips_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print()
print(f"Dataset saved to {OUTPUT_PATH}")
