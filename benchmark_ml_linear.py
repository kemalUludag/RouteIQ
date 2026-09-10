import pandas as pd

from src.ml_data import (
    load_trip_data,
    split_trip_data
)

from src.ml_evaluation import (
    evaluate_regression
)

from src.ml_models import (
    build_linear_regression_pipeline
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

    # ==========================================
    # Build and train model
    # ==========================================

    model = (
        build_linear_regression_pipeline()
    )

    model.fit(
        split.X_train,
        split.y_train
    )

    # ==========================================
    # Validation predictions
    # ==========================================

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
    # Load dummy baseline
    # ==========================================

    baseline_df = pd.read_csv(
        "data/ml_baseline_validation.csv"
    )

    baseline_mae = (
        baseline_df.iloc[0]["mae"]
    )

    mae_improvement_percent = (
        (
            baseline_mae
            - metrics.mae
        )
        /
        baseline_mae
        * 100
    )

    # ==========================================
    # Save result
    # ==========================================

    results_df = pd.DataFrame(
        [
            {
                "model":
                    "LinearRegression",

                "dataset_split":
                    "validation",

                "mae":
                    metrics.mae,

                "rmse":
                    metrics.rmse,

                "r2":
                    metrics.r2,

                "mae_improvement_vs_dummy_percent":
                    mae_improvement_percent,

                "train_samples":
                    len(split.X_train),

                "evaluation_samples":
                    len(split.X_validation)
            }
        ]
    )

    results_df.to_csv(
        "data/ml_linear_validation.csv",
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
        "LINEAR REGRESSION VALIDATION RESULTS"
    )

    print(
        "========================================"
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
        f"\nMAE improvement vs Dummy: "
        f"{mae_improvement_percent:.2f}%"
    )

    print(
        "\nSaved:"
    )

    print(
        "data/ml_linear_validation.csv"
    )

    print(
        "\nTest set remains untouched."
    )


if __name__ == "__main__":
    main()

    