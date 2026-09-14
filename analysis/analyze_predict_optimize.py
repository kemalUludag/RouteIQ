import pandas as pd


def main():

    # ==========================================
    # Load pilot experiment
    # ==========================================

    df = pd.read_csv(
        "data/predict_optimize_results.csv"
    )

    # ==========================================
    # Overall descriptive statistics
    # ==========================================

    overall_summary = {
        "scenarios":
            len(df),

        "networks":
            df[
                "network_seed"
            ].nunique(),

        "mean_edge_mae":
            df[
                "edge_prediction_mae"
            ].mean(),

        "mean_ml_improvement":
            df[
                "ml_improvement_vs_static_percent"
            ].mean(),

        "median_ml_improvement":
            df[
                "ml_improvement_vs_static_percent"
            ].median(),

        "min_ml_improvement":
            df[
                "ml_improvement_vs_static_percent"
            ].min(),

        "max_ml_improvement":
            df[
                "ml_improvement_vs_static_percent"
            ].max(),

        "mean_static_regret":
            df[
                "static_regret_vs_oracle_percent"
            ].mean(),

        "mean_ml_regret":
            df[
                "ml_regret_vs_oracle_percent"
            ].mean(),

        "max_ml_regret":
            df[
                "ml_regret_vs_oracle_percent"
            ].max()
    }

    # ==========================================
    # Environment-level summary
    # ==========================================

    environment_summary = (
        df
        .groupby(
            "environment"
        )
        .agg(
            scenarios=(
                "instance",
                "count"
            ),

            mean_edge_mae=(
                "edge_prediction_mae",
                "mean"
            ),

            mean_ml_improvement=(
                "ml_improvement_vs_static_percent",
                "mean"
            ),

            median_ml_improvement=(
                "ml_improvement_vs_static_percent",
                "median"
            ),

            mean_static_regret=(
                "static_regret_vs_oracle_percent",
                "mean"
            ),

            mean_ml_regret=(
                "ml_regret_vs_oracle_percent",
                "mean"
            ),

            max_ml_regret=(
                "ml_regret_vs_oracle_percent",
                "max"
            )
        )
        .reset_index()
    )

    # ==========================================
    # Network-level summary
    # ==========================================

    network_summary = (
        df
        .groupby(
            "network_seed"
        )
        .agg(
            scenarios=(
                "instance",
                "count"
            ),

            mean_edge_mae=(
                "edge_prediction_mae",
                "mean"
            ),

            mean_ml_improvement=(
                "ml_improvement_vs_static_percent",
                "mean"
            ),

            mean_static_regret=(
                "static_regret_vs_oracle_percent",
                "mean"
            ),

            mean_ml_regret=(
                "ml_regret_vs_oracle_percent",
                "mean"
            ),

            max_ml_regret=(
                "ml_regret_vs_oracle_percent",
                "max"
            )
        )
        .reset_index()
    )

    # ==========================================
    # Nonzero ML regret cases
    # ==========================================

    nonzero_regret = (
        df[
            df[
                "ml_regret_vs_oracle_percent"
            ]
            > 1e-9
        ]
        .copy()
    )

    nonzero_regret = (
        nonzero_regret[
            [
                "instance",
                "environment",
                "edge_prediction_mae",
                "static_true_cost",
                "ml_true_cost",
                "oracle_true_cost",
                "ml_improvement_vs_static_percent",
                "ml_regret_vs_oracle_percent"
            ]
        ]
    )

    # ==========================================
    # Prediction quality vs decision quality
    # ==========================================

    pearson_mae_regret = (
        df[
            [
                "edge_prediction_mae",
                "ml_regret_vs_oracle_percent"
            ]
        ]
        .corr(
            method="pearson"
        )
        .iloc[
            0,
            1
        ]
    )

    spearman_mae_regret = (
        df[
            [
                "edge_prediction_mae",
                "ml_regret_vs_oracle_percent"
            ]
        ]
        .corr(
            method="spearman"
        )
        .iloc[
            0,
            1
        ]
    )

    pearson_mae_improvement = (
        df[
            [
                "edge_prediction_mae",
                "ml_improvement_vs_static_percent"
            ]
        ]
        .corr(
            method="pearson"
        )
        .iloc[
            0,
            1
        ]
    )

    # ==========================================
    # Save analysis
    # ==========================================

    environment_summary.to_csv(
        "data/predict_optimize_environment_summary.csv",
        index=False
    )

    network_summary.to_csv(
        "data/predict_optimize_network_summary.csv",
        index=False
    )

    nonzero_regret.to_csv(
        "data/predict_optimize_nonzero_regret.csv",
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
        "PREDICT → OPTIMIZE PILOT ANALYSIS"
    )

    print(
        "========================================"
    )

    for key, value in overall_summary.items():

        if isinstance(
            value,
            float
        ):
            print(
                f"{key}: "
                f"{value:.4f}"
            )

        else:
            print(
                f"{key}: "
                f"{value}"
            )

    print(
        "\n"
        "========================================"
    )

    print(
        "ENVIRONMENT SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        environment_summary.to_string(
            index=False
        )
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "NETWORK SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        network_summary.to_string(
            index=False
        )
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "NONZERO ML REGRET CASES"
    )

    print(
        "========================================"
    )

    if nonzero_regret.empty:

        print(
            "None"
        )

    else:

        print(
            nonzero_regret.to_string(
                index=False
            )
        )

    print(
        "\n"
        "========================================"
    )

    print(
        "PREDICTION vs DECISION RELATIONSHIP"
    )

    print(
        "========================================"
    )

    print(
        f"Pearson("
        f"edge MAE, ML regret): "
        f"{pearson_mae_regret:.4f}"
    )

    print(
        f"Spearman("
        f"edge MAE, ML regret): "
        f"{spearman_mae_regret:.4f}"
    )

    print(
        f"Pearson("
        f"edge MAE, ML improvement): "
        f"{pearson_mae_improvement:.4f}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This is a descriptive pilot analysis."
    )

    print(
        "Rows are not treated as fully "
        "independent statistical observations."
    )


if __name__ == "__main__":
    main()


    