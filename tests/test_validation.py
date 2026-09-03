from src.validation import validate_solution
from src.models import CVRPInstance
from src.validation import validate_cvrp_routes

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



def test_new_validator_accepts_heterogeneous_solution():
    instance = CVRPInstance(
        name="heterogeneous_test",
        points=[
            (0, 0),
            (1, 0),
            (2, 0)
        ],
        demands=[
            0,
            8,
            4
        ],
        num_vehicles=2,
        vehicle_capacities=[
            5,
            10
        ]
    )

    routes = [
        [0, 2, 0],
        [0, 1, 0]
    ]

    is_valid, message = (
        validate_cvrp_routes(
            instance,
            routes
        )
    )

    assert is_valid is True
    assert message == "Solution is feasible."


def test_new_validator_detects_vehicle_specific_capacity_violation():
    instance = CVRPInstance(
        name="capacity_test",
        points=[
            (0, 0),
            (1, 0),
            (2, 0)
        ],
        demands=[
            0,
            8,
            4
        ],
        num_vehicles=2,
        vehicle_capacities=[
            5,
            10
        ]
    )

    routes = [
        [0, 1, 0],
        [0, 2, 0]
    ]

    is_valid, message = (
        validate_cvrp_routes(
            instance,
            routes
        )
    )

    assert is_valid is False
    assert "capacity" in message.lower()


def test_new_validator_rejects_internal_depot():
    instance = CVRPInstance(
        name="internal_depot_test",
        points=[
            (0, 0),
            (1, 0),
            (2, 0)
        ],
        demands=[
            0,
            2,
            3
        ],
        num_vehicles=1,
        vehicle_capacities=[
            10
        ]
    )

    routes = [
        [0, 1, 0, 2, 0]
    ]

    is_valid, message = (
        validate_cvrp_routes(
            instance,
            routes
        )
    )

    assert is_valid is False
    assert "Depot cannot appear inside" in message

    

       