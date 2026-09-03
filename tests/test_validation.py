from src.validation import validate_solution


demands = [0, 2, 4, 3, 5, 2]
vehicle_capacity = 10
num_customers = 5
num_vehicles = 2


def test_valid_solution():
    routes = [
        [0, 1, 2, 0],
        [0, 3, 4, 5, 0]
    ]

    is_valid, message = validate_solution(
        routes,
        num_customers,
        demands,
        vehicle_capacity,
        num_vehicles
    )

    assert is_valid is True
    assert message == "Solution is feasible."


def test_capacity_violation():
    routes = [
        [0, 2, 3, 4, 0],
        [0, 1, 5, 0]
    ]

    is_valid, message = validate_solution(
        routes,
        num_customers,
        demands,
        vehicle_capacity,
        num_vehicles
    )

    assert is_valid is False
    assert message == "A vehicle capacity constraint is violated."


def test_missing_customer():
    routes = [
        [0, 1, 2, 0],
        [0, 3, 4, 0]
    ]

    is_valid, message = validate_solution(
        routes,
        num_customers,
        demands,
        vehicle_capacity,
        num_vehicles
    )

    assert is_valid is False
    assert message == "Customers are missing or visited more than once."

def test_duplicate_customer():
    routes = [
    [0, 1, 2, 0],
    [0, 2, 3, 5, 0]
]

    is_valid, message = validate_solution(
        routes,
        num_customers,
        demands,
        vehicle_capacity,
        num_vehicles
    )

    assert is_valid is False
    assert message == "Customers are missing or visited more than once."


def test_invalid_depot_start_or_end():
    routes = [
        [1, 2, 0],
        [0, 3, 4, 5, 0]
    ]

    is_valid, message = validate_solution(
        routes,
        num_customers,
        demands,
        vehicle_capacity,
        num_vehicles
    )

    assert is_valid is False
    assert message == "Every route must start and end at the depot."   





       