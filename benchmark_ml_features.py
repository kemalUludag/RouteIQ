import pandas as pd

from src.ml_data import (
    load_trip_data,
    split_trip_data
)

from src.ml_evaluation import (
    evaluate_regression
)

from src.ml_models import (
    build_linear_regression_pipeline,
    build_gradient_boosting_pipeline,
    build_engineered_linear_pipeline,
    build_engineered_gradient_boosting_pipeline
)


def main():

    df = load_trip_data(
        "data/trip_data.csv"
    )

    split = split_trip_data(
        df,
        test_size=0.20,
        validation_size=0.20,
        random_state=42
    )

    models = {
        "LinearOriginal":
            build_linear_regression_pipeline(),

        "LinearEngineered":
            build_engineered_linear_pipeline(),

        "GradientBoostingOriginal":
            build_gradient_boosting_pipeline(),

        "GradientBoostingEngineered":
            build_engineered_gradient_boosting_pipeline()
    }

    results = []

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

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values("mae")
        .reset_index(drop=True)
    )

    results_df.to_csv(
        "data/ml_feature_engineering_validation.csv",
        index=False
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "FEATURE ENGINEERING — VALIDATION"
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
        "\nBest model:"
    )

    print(
        results_df.iloc[0]["model"]
    )

    print(
        "\nTest set remains untouched."
    )


if __name__ == "__main__":
    main()
    