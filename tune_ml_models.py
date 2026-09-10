import json
import time

import pandas as pd

from sklearn.model_selection import (
    RandomizedSearchCV
)

from src.ml_data import (
    load_trip_data,
    split_trip_data
)

from src.ml_evaluation import (
    evaluate_regression
)

from src.ml_models import (
    build_random_forest_pipeline,
    build_engineered_gradient_boosting_pipeline
)


RANDOM_STATE = 42


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
        random_state=RANDOM_STATE
    )

    # ==========================================
    # Random Forest search
    # ==========================================

    print(
        "\n"
        "========================================"
    )

    print(
        "TUNING RANDOM FOREST"
    )

    print(
        "========================================"
    )

    random_forest = (
        build_random_forest_pipeline()
    )

    # Avoid nested parallelism:
    # RandomizedSearchCV handles parallel jobs.
    random_forest.set_params(
        model__n_jobs=1
    )

    random_forest_parameters = {
        "model__n_estimators": [
            200,
            400,
            600
        ],

        "model__max_depth": [
            None,
            8,
            12,
            16
        ],

        "model__min_samples_split": [
            2,
            5,
            10
        ],

        "model__min_samples_leaf": [
            1,
            2,
            4
        ],

        "model__max_features": [
            1.0,
            "sqrt"
        ]
    }

    rf_search = RandomizedSearchCV(
        estimator=random_forest,
        param_distributions=(
            random_forest_parameters
        ),
        n_iter=15,
        scoring=(
            "neg_mean_absolute_error"
        ),
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        return_train_score=False
    )

    rf_start = (
        time.perf_counter()
    )

    rf_search.fit(
        split.X_train,
        split.y_train
    )

    rf_tuning_runtime = (
        time.perf_counter()
        - rf_start
    )

    rf_best_model = (
        rf_search.best_estimator_
    )

    rf_validation_predictions = (
        rf_best_model.predict(
            split.X_validation
        )
    )

    rf_validation_metrics = (
        evaluate_regression(
            split.y_validation,
            rf_validation_predictions
        )
    )

    # ==========================================
    # Gradient Boosting search
    # ==========================================

    print(
        "\n"
        "========================================"
    )

    print(
        "TUNING GRADIENT BOOSTING"
    )

    print(
        "========================================"
    )

    gradient_boosting = (
        build_engineered_gradient_boosting_pipeline()
    )

    gradient_boosting_parameters = {
        "model__n_estimators": [
            150,
            250,
            350,
            500
        ],

        "model__learning_rate": [
            0.02,
            0.03,
            0.05,
            0.08,
            0.10
        ],

        "model__max_depth": [
            2,
            3,
            4
        ],

        "model__min_samples_split": [
            2,
            5,
            10
        ],

        "model__min_samples_leaf": [
            1,
            2,
            4,
            8
        ],

        "model__subsample": [
            0.7,
            0.85,
            1.0
        ]
    }

    gb_search = RandomizedSearchCV(
        estimator=gradient_boosting,
        param_distributions=(
            gradient_boosting_parameters
        ),
        n_iter=20,
        scoring=(
            "neg_mean_absolute_error"
        ),
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        return_train_score=False
    )

    gb_start = (
        time.perf_counter()
    )

    gb_search.fit(
        split.X_train,
        split.y_train
    )

    gb_tuning_runtime = (
        time.perf_counter()
        - gb_start
    )

    gb_best_model = (
        gb_search.best_estimator_
    )

    gb_validation_predictions = (
        gb_best_model.predict(
            split.X_validation
        )
    )

    gb_validation_metrics = (
        evaluate_regression(
            split.y_validation,
            gb_validation_predictions
        )
    )

    # ==========================================
    # Validation comparison
    # ==========================================

    results_df = pd.DataFrame(
        [
            {
                "model":
                    "RandomForestTuned",

                "cv_mae":
                    -rf_search.best_score_,

                "validation_mae":
                    rf_validation_metrics.mae,

                "validation_rmse":
                    rf_validation_metrics.rmse,

                "validation_r2":
                    rf_validation_metrics.r2,

                "tuning_runtime_seconds":
                    rf_tuning_runtime
            },

            {
                "model":
                    "GradientBoostingTuned",

                "cv_mae":
                    -gb_search.best_score_,

                "validation_mae":
                    gb_validation_metrics.mae,

                "validation_rmse":
                    gb_validation_metrics.rmse,

                "validation_r2":
                    gb_validation_metrics.r2,

                "tuning_runtime_seconds":
                    gb_tuning_runtime
            }
        ]
    )

    results_df = (
        results_df
        .sort_values(
            "validation_mae"
        )
        .reset_index(
            drop=True
        )
    )

    results_df.to_csv(
        "data/ml_tuning_validation.csv",
        index=False
    )

    # ==========================================
    # Save best parameters
    # ==========================================

    best_parameters = {
        "RandomForestTuned":
            rf_search.best_params_,

        "GradientBoostingTuned":
            gb_search.best_params_
    }

    with open(
        "data/ml_best_parameters.json",
        "w"
    ) as file:
        json.dump(
            best_parameters,
            file,
            indent=4
        )

    # ==========================================
    # Save detailed CV search results
    # ==========================================

    rf_cv_results = pd.DataFrame(
        rf_search.cv_results_
    )

    rf_cv_results.to_csv(
        "data/ml_rf_cv_results.csv",
        index=False
    )

    gb_cv_results = pd.DataFrame(
        gb_search.cv_results_
    )

    gb_cv_results.to_csv(
        "data/ml_gb_cv_results.csv",
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
        "TUNED MODEL VALIDATION RESULTS"
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
        "\nBest Random Forest parameters:"
    )

    print(
        rf_search.best_params_
    )

    print(
        "\nBest Gradient Boosting parameters:"
    )

    print(
        gb_search.best_params_
    )

    print(
        "\nBest validation model:"
    )

    print(
        results_df.iloc[0]["model"]
    )

    print(
        "\nSaved:"
    )

    print(
        "data/ml_tuning_validation.csv"
    )

    print(
        "data/ml_best_parameters.json"
    )

    print(
        "data/ml_rf_cv_results.csv"
    )

    print(
        "data/ml_gb_cv_results.csv"
    )

    print(
        "\nTest set remains untouched."
    )


if __name__ == "__main__":
    main()


    