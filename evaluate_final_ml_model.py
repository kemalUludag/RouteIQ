import pandas as pd

from src.ml_data import (
    load_trip_data,
    split_trip_data
)

from src.ml_evaluation import (
    evaluate_regression
)

from src.ml_models import (
    build_engineered_gradient_boosting_pipeline
)


RANDOM_STATE = 42


FINAL_PARAMETERS = {
    "model__subsample": 0.7,
    "model__n_estimators": 250,
    "model__min_samples_split": 10,
    "model__min_samples_leaf": 2,
    "model__max_depth": 4,
    "model__learning_rate": 0.05
}


def main():

    # ==========================================
    # Load original dataset
    # ==========================================

    df = load_trip_data(
        "data/trip_data.csv"
    )

    # ==========================================
    # Reproduce original split
    # ==========================================

    split = split_trip_data(
        df,
        test_size=0.20,
        validation_size=0.20,
        random_state=RANDOM_STATE
    )

    # ==========================================
    # Model selection is finished.
    #
    # We can now combine train + validation
    # and use the full 80% development data
    # for final training.
    # ==========================================

    X_development = pd.concat(
        [
            split.X_train,
            split.X_validation
        ],
        axis=0
    )

    y_development = pd.concat(
        [
            split.y_train,
            split.y_validation
        ],
        axis=0
    )

    # ==========================================
    # Build final selected model
    # ==========================================

    model = (
        build_engineered_gradient_boosting_pipeline()
    )

    model.set_params(
        **FINAL_PARAMETERS
    )

    # ==========================================
    # Final training
    # ==========================================

    print(
        "Training final model on "
        "train + validation data..."
    )

    model.fit(
        X_development,
        y_development
    )

    # ==========================================
    # FIRST evaluation on locked test set
    # ==========================================

    print(
        "Evaluating locked test set..."
    )

    test_predictions = model.predict(
        split.X_test
    )

    metrics = evaluate_regression(
        split.y_test,
        test_predictions
    )

    # ==========================================
    # Save final test metrics
    # ==========================================

    results_df = pd.DataFrame(
        [
            {
                "model":
                    "GradientBoostingTuned",

                "train_samples":
                    len(X_development),

                "test_samples":
                    len(split.X_test),

                "test_mae":
                    metrics.mae,

                "test_rmse":
                    metrics.rmse,

                "test_r2":
                    metrics.r2
            }
        ]
    )

    results_df.to_csv(
        "data/ml_final_test_results.csv",
        index=False
    )

    # ==========================================
    # Save test predictions for later
    # error analysis / downstream evaluation
    # ==========================================

    predictions_df = (
        split.X_test.copy()
    )

    predictions_df[
        "actual_travel_time"
    ] = split.y_test.to_numpy()

    predictions_df[
        "predicted_travel_time"
    ] = test_predictions

    predictions_df[
        "error"
    ] = (
        predictions_df[
            "predicted_travel_time"
        ]
        -
        predictions_df[
            "actual_travel_time"
        ]
    )

    predictions_df[
        "absolute_error"
    ] = (
        predictions_df[
            "error"
        ].abs()
    )

    predictions_df.to_csv(
        "data/ml_final_test_predictions.csv",
        index=False
    )

    # ==========================================
    # Compare against original dummy baseline
    # ==========================================

    baseline_df = pd.read_csv(
        "data/ml_baseline_validation.csv"
    )

    baseline_mae = (
        baseline_df.iloc[0]["mae"]
    )

    approximate_improvement = (
        (
            baseline_mae
            - metrics.mae
        )
        /
        baseline_mae
        * 100
    )

    # ==========================================
    # Report
    # ==========================================

    print(
        "\n"
        "========================================"
    )

    print(
        "FINAL LOCKED TEST RESULTS"
    )

    print(
        "========================================"
    )

    print(
        "Model: GradientBoostingTuned"
    )

    print(
        f"Development samples: "
        f"{len(X_development)}"
    )

    print(
        f"Test samples: "
        f"{len(split.X_test)}"
    )

    print(
        f"\nTest MAE:  "
        f"{metrics.mae:.3f} minutes"
    )

    print(
        f"Test RMSE: "
        f"{metrics.rmse:.3f} minutes"
    )

    print(
        f"Test R²:   "
        f"{metrics.r2:.4f}"
    )

    print(
        f"\nApprox. MAE improvement "
        f"vs original Dummy baseline: "
        f"{approximate_improvement:.2f}%"
    )

    print(
        "\nSaved:"
    )

    print(
        "data/ml_final_test_results.csv"
    )

    print(
        "data/ml_final_test_predictions.csv"
    )

    print(
        "\nFinal test evaluation complete."
    )

    print(
        "Do not tune the model using "
        "these test results."
    )


if __name__ == "__main__":
    main()

    