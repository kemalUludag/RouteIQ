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
    build_engineered_linear_pipeline,
    build_random_forest_pipeline,
    build_engineered_gradient_boosting_pipeline
)


RANDOM_STATE = 42


RF_FINAL_PARAMETERS = {
    "model__n_estimators": 200,
    "model__min_samples_split": 5,
    "model__min_samples_leaf": 1,
    "model__max_features": 1.0,
    "model__max_depth": 12,
    "model__n_jobs": -1
}


GB_FINAL_PARAMETERS = {
    "model__subsample": 0.7,
    "model__n_estimators": 250,
    "model__min_samples_split": 10,
    "model__min_samples_leaf": 2,
    "model__max_depth": 4,
    "model__learning_rate": 0.05
}


def main():

    # ==========================================
    # Load and reproduce locked split
    # ==========================================

    df = load_trip_data(
        "data/trip_data.csv"
    )

    split = split_trip_data(
        df,
        test_size=0.20,
        validation_size=0.20,
        random_state=RANDOM_STATE
    )

    # ==========================================
    # Final development set = train + validation
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
    # Finalized models
    # ==========================================

    dummy_model = DummyRegressor(
        strategy="mean"
    )

    linear_model = (
        build_engineered_linear_pipeline()
    )

    random_forest_model = (
        build_random_forest_pipeline()
    )

    random_forest_model.set_params(
        **RF_FINAL_PARAMETERS
    )

    gradient_boosting_model = (
        build_engineered_gradient_boosting_pipeline()
    )

    gradient_boosting_model.set_params(
        **GB_FINAL_PARAMETERS
    )

    models = {
        "DummyMean":
            dummy_model,

        "LinearRegressionEngineered":
            linear_model,

        "RandomForestTuned":
            random_forest_model,

        "GradientBoostingTuned":
            gradient_boosting_model
    }

    results = []

    prediction_frames = []

    # ==========================================
    # Train on same 8000 development samples
    # and evaluate on same locked 2000 test rows
    # ==========================================

    for model_name, model in models.items():

        print(
            f"Evaluating {model_name}..."
        )

        model.fit(
            X_development,
            y_development
        )

        predictions = model.predict(
            split.X_test
        )

        metrics = evaluate_regression(
            split.y_test,
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
                    metrics.r2,

                "development_samples":
                    len(X_development),

                "test_samples":
                    len(split.X_test)
            }
        )

        prediction_df = (
            split.X_test.copy()
        )

        prediction_df[
            "actual_travel_time"
        ] = split.y_test.to_numpy()

        prediction_df[
            "predicted_travel_time"
        ] = predictions

        prediction_df[
            "error"
        ] = (
            prediction_df[
                "predicted_travel_time"
            ]
            -
            prediction_df[
                "actual_travel_time"
            ]
        )

        prediction_df[
            "absolute_error"
        ] = (
            prediction_df[
                "error"
            ].abs()
        )

        prediction_df[
            "model"
        ] = model_name

        prediction_frames.append(
            prediction_df
        )

    # ==========================================
    # Comparison table
    # ==========================================

    results_df = pd.DataFrame(
        results
    )

    dummy_mae = (
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
            dummy_mae
            - results_df["mae"]
        )
        /
        dummy_mae
        * 100
    )

    results_df = (
        results_df
        .sort_values(
            "mae"
        )
        .reset_index(
            drop=True
        )
    )

    results_df.to_csv(
        "data/ml_final_model_comparison.csv",
        index=False
    )

    # ==========================================
    # Save all test predictions
    # ==========================================

    all_predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True
    )

    all_predictions_df.to_csv(
        "data/ml_final_model_predictions.csv",
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
        "FINAL ML TEST BENCHMARK"
    )

    print(
        "========================================"
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        "\nSelected production candidate:"
    )

    print(
        "GradientBoostingTuned"
    )

    print(
        "\nNo further model selection or "
        "hyperparameter tuning should be "
        "performed using this test set."
    )


if __name__ == "__main__":
    main()

    