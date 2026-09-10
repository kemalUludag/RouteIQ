import pandas as pd

from src.ml_data import (
    load_trip_data,
    split_trip_data
)

from src.ml_models import (
    build_random_forest_pipeline,
    build_gradient_boosting_pipeline
)


def create_prediction_frame(
    model_name,
    model,
    X_train,
    y_train,
    X_validation,
    y_validation
):
    # ==========================================
    # Train model only on training data
    # ==========================================

    model.fit(
        X_train,
        y_train
    )

    # ==========================================
    # Predict validation data
    # ==========================================

    predictions = model.predict(
        X_validation
    )

    # ==========================================
    # Build detailed prediction table
    # ==========================================

    result = X_validation.copy()

    result["actual_travel_time"] = (
        y_validation.to_numpy()
    )

    result["predicted_travel_time"] = (
        predictions
    )

    # Positive error:
    # model predicted too much travel time
    #
    # Negative error:
    # model predicted too little travel time

    result["error"] = (
        result["predicted_travel_time"]
        - result["actual_travel_time"]
    )

    result["absolute_error"] = (
        result["error"].abs()
    )

    result["model"] = (
        model_name
    )

    return result


def main():

    # ==========================================
    # Load data
    # ==========================================

    df = load_trip_data(
        "data/trip_data.csv"
    )

    # ==========================================
    # Use the exact same split as before
    # ==========================================

    split = split_trip_data(
        df,
        test_size=0.20,
        validation_size=0.20,
        random_state=42
    )

    # ==========================================
    # Models to analyze
    # ==========================================

    models = {
        "RandomForest":
            build_random_forest_pipeline(),

        "GradientBoosting":
            build_gradient_boosting_pipeline()
    }

    prediction_frames = []

    # ==========================================
    # Train and create prediction tables
    # ==========================================

    for model_name, model in models.items():

        print(
            f"Analyzing {model_name}..."
        )

        prediction_frame = (
            create_prediction_frame(
                model_name=model_name,
                model=model,
                X_train=split.X_train,
                y_train=split.y_train,
                X_validation=split.X_validation,
                y_validation=split.y_validation
            )
        )

        prediction_frames.append(
            prediction_frame
        )

    # ==========================================
    # Combine both models
    # ==========================================

    results_df = pd.concat(
        prediction_frames,
        ignore_index=True
    )

    # ==========================================
    # Save individual predictions
    # ==========================================

    results_df.to_csv(
        "data/ml_validation_predictions.csv",
        index=False
    )

    # ==========================================
    # Error analysis by road type
    # ==========================================

    road_summary = (
        results_df
        .groupby(
            [
                "model",
                "road_type"
            ]
        )
        .agg(
            samples=(
                "absolute_error",
                "count"
            ),

            mae=(
                "absolute_error",
                "mean"
            ),

            mean_error=(
                "error",
                "mean"
            )
        )
        .reset_index()
    )

    road_summary.to_csv(
        "data/ml_error_by_road_type.csv",
        index=False
    )

    # ==========================================
    # Error analysis by weather
    # ==========================================

    weather_summary = (
        results_df
        .groupby(
            [
                "model",
                "weather"
            ]
        )
        .agg(
            samples=(
                "absolute_error",
                "count"
            ),

            mae=(
                "absolute_error",
                "mean"
            ),

            mean_error=(
                "error",
                "mean"
            )
        )
        .reset_index()
    )

    weather_summary.to_csv(
        "data/ml_error_by_weather.csv",
        index=False
    )

    # ==========================================
    # Error analysis by traffic level
    # ==========================================

    traffic_summary = (
        results_df
        .groupby(
            [
                "model",
                "traffic_level"
            ]
        )
        .agg(
            samples=(
                "absolute_error",
                "count"
            ),

            mae=(
                "absolute_error",
                "mean"
            ),

            mean_error=(
                "error",
                "mean"
            )
        )
        .reset_index()
    )

    traffic_summary.to_csv(
        "data/ml_error_by_traffic.csv",
        index=False
    )

    # ==========================================
    # Create distance bands
    # ==========================================

    results_df["distance_band"] = (
        pd.cut(
            results_df["distance"],
            bins=[
                0,
                10,
                20,
                30,
                40
            ],
            include_lowest=True
        )
    )

    # ==========================================
    # Error analysis by distance
    # ==========================================

    distance_summary = (
        results_df
        .groupby(
            [
                "model",
                "distance_band"
            ],
            observed=True
        )
        .agg(
            samples=(
                "absolute_error",
                "count"
            ),

            mae=(
                "absolute_error",
                "mean"
            ),

            mean_error=(
                "error",
                "mean"
            )
        )
        .reset_index()
    )

    distance_summary.to_csv(
        "data/ml_error_by_distance.csv",
        index=False
    )

    # ==========================================
    # Print reports
    # ==========================================

    print(
        "\n"
        "========================================"
    )

    print(
        "ERROR BY ROAD TYPE"
    )

    print(
        "========================================"
    )

    print(
        road_summary.to_string(
            index=False
        )
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "ERROR BY WEATHER"
    )

    print(
        "========================================"
    )

    print(
        weather_summary.to_string(
            index=False
        )
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "ERROR BY TRAFFIC LEVEL"
    )

    print(
        "========================================"
    )

    print(
        traffic_summary.to_string(
            index=False
        )
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "ERROR BY DISTANCE"
    )

    print(
        "========================================"
    )

    print(
        distance_summary.to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )

    print(
        "data/ml_validation_predictions.csv"
    )

    print(
        "data/ml_error_by_road_type.csv"
    )

    print(
        "data/ml_error_by_weather.csv"
    )

    print(
        "data/ml_error_by_traffic.csv"
    )

    print(
        "data/ml_error_by_distance.csv"
    )

    print(
        "\nTest set remains untouched."
    )


if __name__ == "__main__":
    main()

    