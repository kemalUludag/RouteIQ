import numpy as np
import pandas as pd


def add_engineered_features(
    X: pd.DataFrame
) -> pd.DataFrame:

    result = X.copy()

    # ==========================================
    # Distance × traffic interaction
    # ==========================================

    result[
        "distance_x_traffic"
    ] = (
        result["distance"]
        * result["traffic_level"]
    )

    # ==========================================
    # Cyclical hour representation
    # ==========================================

    result[
        "hour_sin"
    ] = np.sin(
        2
        * np.pi
        * result["hour"]
        / 24
    )

    result[
        "hour_cos"
    ] = np.cos(
        2
        * np.pi
        * result["hour"]
        / 24
    )

    # ==========================================
    # Weekend indicator
    # ==========================================

    result[
        "is_weekend"
    ] = (
        result["weekday"] >= 5
    ).astype(int)

    return result
