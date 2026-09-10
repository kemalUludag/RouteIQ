import pandas as pd

from sklearn.dummy import (
    DummyRegressor
)

from src.ml_data import (
    load_trip_data,
    split_trip_data
)

from src.ml_evaluation import (
    evaluate_regression
)


def main():
    # ==========================================
    # Load and split data
    # ==========================================

    df = load_trip_data(
        "data/trip_data.csv"
    )

    split = split_trip_data(
        df,
        test_size=0.20,
        validation_size=0.20,
        random_state=42
    )

    print(
        "Dataset size:",
        len(df)
    )

    print(
        "Train size:",
        len(split.X_train)
    )

    print(
        "Validation size:",
        len(split.X_validation)
    )

    print(
        "Test size:",
        len(split.X_test)
    )

    # ==========================================
    # Dummy baseline
    # ==========================================

    model = DummyRegressor(
        strategy="mean"
    )

    model.fit(
        split.X_train,
        split.y_train
    )

    validation_predictions = (
        model.predict(
            split.X_validation
        )
    )

    metrics = evaluate_regression(
        split.y_validation,
        validation_predictions
    )

    # ==========================================
    # Save benchmark result
    # ==========================================

    results_df = pd.DataFrame(
        [
            {
                "model":
                    "DummyMean",

                "dataset_split":
                    "validation",

                "mae":
                    metrics.mae,

                "rmse":
                    metrics.rmse,

                "r2":
                    metrics.r2,

                "train_samples":
                    len(
                        split.X_train
                    ),

                "evaluation_samples":
                    len(
                        split.X_validation
                    )
            }
        ]
    )

    results_df.to_csv(
        "data/ml_baseline_validation.csv",
        index=False
    )

    # ==========================================
    # Report
    # ==========================================

    print(
        "\n"
        "========================================"
    )

    print(
        "ML BASELINE VALIDATION RESULTS"
    )

    print(
        "========================================"
    )

    print(
        f"Prediction strategy: "
        f"training-set mean"
    )

    print(
        f"Predicted travel time: "
        f"{model.constant_.item():.3f} minutes"
    )

    print(
        f"MAE:  "
        f"{metrics.mae:.3f} minutes"
    )

    print(
        f"RMSE: "
        f"{metrics.rmse:.3f} minutes"
    )

    print(
        f"R²:   "
        f"{metrics.r2:.4f}"
    )

    print(
        "\nSaved:"
    )

    print(
        "data/ml_baseline_validation.csv"
    )

    print(
        "\nTest set remains untouched."
    )


if __name__ == "__main__":
    main()
    