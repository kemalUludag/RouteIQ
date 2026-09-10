from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder
)

from src.ml_features import (
    add_engineered_features
)


BASE_NUMERIC_FEATURES = [
    "distance",
    "hour",
    "weekday",
    "traffic_level"
]


ENGINEERED_NUMERIC_FEATURES = [
    "distance",
    "hour",
    "weekday",
    "traffic_level",
    "distance_x_traffic",
    "hour_sin",
    "hour_cos",
    "is_weekend"
]


CATEGORICAL_FEATURES = [
    "road_type",
    "weather"
]


def build_preprocessor(
    numeric_features
):
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                numeric_features
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                CATEGORICAL_FEATURES
            )
        ]
    )


# ==============================================
# Original models
# ==============================================

def build_linear_regression_pipeline():
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    BASE_NUMERIC_FEATURES
                )
            ),
            (
                "model",
                LinearRegression()
            )
        ]
    )


def build_random_forest_pipeline():
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    BASE_NUMERIC_FEATURES
                )
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )


def build_gradient_boosting_pipeline():
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    BASE_NUMERIC_FEATURES
                )
            ),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42
                )
            )
        ]
    )


# ==============================================
# Feature-engineered models
# ==============================================

def build_engineered_linear_pipeline():
    return Pipeline(
        steps=[
            (
                "feature_engineering",
                FunctionTransformer(
                    add_engineered_features,
                    validate=False
                )
            ),
            (
                "preprocessor",
                build_preprocessor(
                    ENGINEERED_NUMERIC_FEATURES
                )
            ),
            (
                "model",
                LinearRegression()
            )
        ]
    )


def build_engineered_gradient_boosting_pipeline():
    return Pipeline(
        steps=[
            (
                "feature_engineering",
                FunctionTransformer(
                    add_engineered_features,
                    validate=False
                )
            ),
            (
                "preprocessor",
                build_preprocessor(
                    ENGINEERED_NUMERIC_FEATURES
                )
            ),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42
                )
            )
        ]
    )

