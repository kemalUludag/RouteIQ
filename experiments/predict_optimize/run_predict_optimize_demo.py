import numpy as np
import pandas as pd

from routeiq.data_generator import (
    generate_cvrp_instance_model
)

from routeiq.distance import (
    create_distance_matrix
)

from routeiq.ml_training import (
    train_final_travel_time_model
)

from routeiq.ortools_solver import (
    ORToolsCVRPSolver
)

from routeiq.routing import (
    calculate_routes_cost
)

from routeiq.travel_time_matrix import (
    generate_routing_context,
    build_predicted_travel_time_matrix,
    build_true_expected_travel_time_matrix
)


def calculate_matrix_mae(
    predicted_matrix,
    true_matrix
):
    predicted_values = []
    true_values = []

    num_nodes = len(
        true_matrix
    )

    for i in range(num_nodes):
        for j in range(num_nodes):

            if i == j:
                continue

            predicted_values.append(
                predicted_matrix[i][j]
            )

            true_values.append(
                true_matrix[i][j]
            )

    predicted_values = np.asarray(
        predicted_values
    )

    true_values = np.asarray(
        true_values
    )

    return float(
        np.mean(
            np.abs(
                predicted_values
                - true_values
            )
        )
    )


def main():

    # ==========================================
    # 1. Train finalized travel-time predictor
    # ==========================================

    print(
        "Training final travel-time model..."
    )

    model = (
        train_final_travel_time_model()
    )

    # ==========================================
    # 2. Generate one CVRP instance
    # ==========================================

    instance = (
        generate_cvrp_instance_model(
            num_customers=10,
            num_vehicles=2,
            vehicle_capacity=15,
            seed=7
        )
    )

    # ==========================================
    # 3. Generate routing environment
    #
    # Wednesday, 18:00, rainy
    # ==========================================

    scenario = (
        generate_routing_context(
            instance=instance,
            hour=18,
            weekday=2,
            weather="rain",
            seed=123
        )
    )

    # ==========================================
    # 4. Build the three cost representations
    # ==========================================

    distance_matrix = (
        create_distance_matrix(
            instance.points
        )
    )

    predicted_matrix = (
        build_predicted_travel_time_matrix(
            instance=instance,
            scenario=scenario,
            model=model
        )
    )

    true_matrix = (
        build_true_expected_travel_time_matrix(
            instance=instance,
            scenario=scenario
        )
    )

    # ==========================================
    # 5. Measure edge-level prediction quality
    #
    # This is NOT yet routing quality.
    # It only measures prediction error.
    # ==========================================

    matrix_mae = (
        calculate_matrix_mae(
            predicted_matrix,
            true_matrix
        )
    )

    # ==========================================
    # 6. Use exactly the same optimization
    # algorithm for all three decisions.
    #
    # This controls for solver differences.
    # ==========================================

    solver = ORToolsCVRPSolver(
        time_limit_seconds=1
    )

    # ------------------------------------------
    # Static routing:
    # optimize Euclidean distance
    # ------------------------------------------

    static_solution = (
        solver.solve(
            instance
        )
    )

    # ------------------------------------------
    # ML-assisted routing:
    # optimize predicted travel time
    # ------------------------------------------

    ml_solution = (
        solver.solve_with_cost_matrix(
            instance=instance,
            cost_matrix=predicted_matrix,
            objective_name=(
                "predicted_travel_time"
            ),
            objective_unit="minutes"
        )
    )

    # ------------------------------------------
    # Oracle-informed benchmark:
    # optimize true expected travel time
    #
    # IMPORTANT:
    # This is not guaranteed to be globally
    # optimal because OR-Tools is still the
    # optimization engine.
    # ------------------------------------------

    oracle_solution = (
        solver.solve_with_cost_matrix(
            instance=instance,
            cost_matrix=true_matrix,
            objective_name=(
                "true_expected_travel_time"
            ),
            objective_unit="minutes"
        )
    )

    # ==========================================
    # 7. Safety check
    # ==========================================

    solutions = {
        "StaticDistance":
            static_solution,

        "MLPredicted":
            ml_solution,

        "OracleInformed":
            oracle_solution
    }

    for name, solution in solutions.items():

        if not solution.feasible:
            raise RuntimeError(
                f"{name} did not produce "
                f"a feasible solution."
            )

    # ==========================================
    # 8. FAIR evaluation
    #
    # Critical point:
    #
    # We do NOT compare each solver's own
    # objective value.
    #
    # Every route is evaluated using the SAME
    # true expected travel-time matrix.
    # ==========================================

    static_true_cost = (
        calculate_routes_cost(
            static_solution.routes,
            true_matrix
        )
    )

    ml_true_cost = (
        calculate_routes_cost(
            ml_solution.routes,
            true_matrix
        )
    )

    oracle_true_cost = (
        calculate_routes_cost(
            oracle_solution.routes,
            true_matrix
        )
    )

    # ==========================================
    # 9. Decision-quality metrics
    # ==========================================

    ml_improvement_vs_static = (
        (
            static_true_cost
            - ml_true_cost
        )
        /
        static_true_cost
        * 100
    )

    ml_regret_vs_oracle = (
        (
            ml_true_cost
            - oracle_true_cost
        )
        /
        oracle_true_cost
        * 100
    )

    static_regret_vs_oracle = (
        (
            static_true_cost
            - oracle_true_cost
        )
        /
        oracle_true_cost
        * 100
    )

    # ==========================================
    # 10. Save summary
    # ==========================================

    results_df = pd.DataFrame(
        [
            {
                "strategy":
                    "StaticDistance",

                "true_expected_cost":
                    static_true_cost,

                "regret_vs_oracle_percent":
                    static_regret_vs_oracle
            },

            {
                "strategy":
                    "MLPredicted",

                "true_expected_cost":
                    ml_true_cost,

                "regret_vs_oracle_percent":
                    ml_regret_vs_oracle
            },

            {
                "strategy":
                    "OracleInformed",

                "true_expected_cost":
                    oracle_true_cost,

                "regret_vs_oracle_percent":
                    0.0
            }
        ]
    )

    results_df.to_csv(
        "data/predict_optimize_demo.csv",
        index=False
    )

    # ==========================================
    # 11. Report
    # ==========================================

    print(
        "\n"
        "========================================"
    )

    print(
        "PREDICT → OPTIMIZE DEMO"
    )

    print(
        "========================================"
    )

    print(
        f"Instance: "
        f"{instance.name}"
    )

    print(
        f"Customers: "
        f"{instance.num_customers}"
    )

    print(
        f"Vehicles: "
        f"{instance.num_vehicles}"
    )

    print(
        "\nScenario:"
    )

    print(
        f"Hour:    {scenario.hour}"
    )

    print(
        f"Weekday: {scenario.weekday}"
    )

    print(
        f"Weather: {scenario.weather}"
    )

    print(
        f"\nEdge-level prediction MAE: "
        f"{matrix_mae:.3f} minutes"
    )

    print(
        "\n----------------------------------------"
    )

    print(
        "ROUTES"
    )

    print(
        "----------------------------------------"
    )

    print(
        "Static distance route:"
    )

    print(
        static_solution.routes
    )

    print(
        "\nML-predicted route:"
    )

    print(
        ml_solution.routes
    )

    print(
        "\nOracle-informed route:"
    )

    print(
        oracle_solution.routes
    )

    print(
        "\n----------------------------------------"
    )

    print(
        "TRUE EXPECTED TRAVEL-TIME EVALUATION"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"Static route: "
        f"{static_true_cost:.3f} minutes"
    )

    print(
        f"ML route:     "
        f"{ml_true_cost:.3f} minutes"
    )

    print(
        f"Oracle route: "
        f"{oracle_true_cost:.3f} minutes"
    )

    print(
        "\n----------------------------------------"
    )

    print(
        "DECISION QUALITY"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"ML improvement vs static: "
        f"{ml_improvement_vs_static:.2f}%"
    )

    print(
        f"Static regret vs oracle:   "
        f"{static_regret_vs_oracle:.2f}%"
    )

    print(
        f"ML regret vs oracle:       "
        f"{ml_regret_vs_oracle:.2f}%"
    )

    print(
        "\nSaved:"
    )

    print(
        "data/predict_optimize_demo.csv"
    )


if __name__ == "__main__":
    main()

    