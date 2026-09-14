import pandas as pd

from .ml_data import (
    load_trip_data,
    split_trip_data
)

from .ml_models import (
    build_final_gradient_boosting_pipeline
)


def train_final_travel_time_model(
    filepath="data/trip_data.csv",
    random_state=42
):
    # ==========================================
    # Load data
    # ==========================================

    df = load_trip_data(
        filepath
    )

    # ==========================================
    # Reproduce the locked development/test split
    # ==========================================

    split = split_trip_data(
        df,
        test_size=0.20,
        validation_size=0.20,
        random_state=random_state
    )

    # ==========================================
    # Model selection is already complete.
    #
    # Train + validation are combined:
    #
    # 6000 + 2000 = 8000 observations
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
        build_final_gradient_boosting_pipeline()
    )

    # ==========================================
    # Train
    # ==========================================

    model.fit(
        X_development,
        y_development
    )

    return model

