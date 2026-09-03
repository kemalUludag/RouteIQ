from src.trip_data_generator import generate_trip_dataset


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
    "data/trip_data.csv",
    index=False
)

print()
print("Dataset saved to data/trip_data.csv")
