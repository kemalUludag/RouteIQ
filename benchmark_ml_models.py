import pandas as pd

from sklearn.dummy import DummyRegressor

from src.ml_data import (
    load_trip_data,
    split_trip_data
)

from src.ml_evaluation import (
    evaluate_regression
)

from src.ml_models import (
    build_linear_regression_pipeline,
    build_random_forest_pipeline,
    build_gradient_boosting_pipeline
)


def main():
    # ==========================================
    # Load and split
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
    # Candidate models
    # ==========================================

    models = {
        "DummyMean":
            DummyRegressor(
                strategy="mean"
            ),

        "LinearRegression":
            build_linear_regression_pipeline(),

        "RandomForest":
            build_random_forest_pipeline(),

        "GradientBoosting":
            build_gradient_boosting_pipeline()
    }

    results = []

    # ==========================================
    # Train and evaluate
    # ==========================================

    for model_name, model in models.items():

        print(
            f"Training {model_name}..."
        )

        model.fit(
            split.X_train,
            split.y_train
        )

        predictions = model.predict(
            split.X_validation
        )

        metrics = evaluate_regression(
            split.y_validation,
            predictions
        )

        results.append(
            {
                "model":
                    model_name,

                "mae":
                    metrics.mae,

                "rmse":
                    metrics.rmse,

                "r2":
                    metrics.r2
            }
        )

    # ==========================================
    # Results
    # ==========================================

    results_df = pd.DataFrame(
        results
    )

    baseline_mae = (
        results_df.loc[
            results_df["model"]
            == "DummyMean",
            "mae"
        ].iloc[0]
    )

    results_df[
        "mae_improvement_vs_dummy_percent"
    ] = (
        (
            baseline_mae
            - results_df["mae"]
        )
        /
        baseline_mae
        * 100
    )

    results_df = (
        results_df
        .sort_values("mae")
        .reset_index(drop=True)
    )

    results_df.to_csv(
        "data/ml_model_comparison_validation.csv",
        index=False
    )

    # ==========================================
    # Print report
    # ==========================================

    print(
        "\n"
        "========================================"
    )

    print(
        "ML MODEL COMPARISON — VALIDATION"
    )

    print(
        "========================================"
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    best_model = (
        results_df.iloc[0]
    )

    print(
        "\nBest validation model:"
    )

    print(
        best_model["model"]
    )

    print(
        f"MAE: "
        f"{best_model['mae']:.3f} minutes"
    )

    print(
        f"R²: "
        f"{best_model['r2']:.4f}"
    )

    print(
        "\nTest set remains untouched."
    )


if __name__ == "__main__":
    main()
    