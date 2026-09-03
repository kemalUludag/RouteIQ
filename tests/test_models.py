import pytest

from src.models import CVRPInstance
from src.problem_data import create_default_instance

def test_valid_cvrp_instance():
    instance = CVRPInstance(
        name="test_instance",
        points=[
            (0, 0),
            (2, 3),
            (5, 4)
        ],
        demands=[
            0,
            2,
            4
        ],
        num_vehicles=2,
        vehicle_capacities=[
            10,
            10
        ],
        depot_index=0
    )

    assert instance.name == "test_instance"
    assert instance.num_nodes == 3
    assert instance.num_customers == 2
    assert instance.demands == [0, 2, 4]


def test_points_and_demands_must_match():
    with pytest.raises(
        ValueError,
        match="The number of points must match"
    ):
        CVRPInstance(
            name="invalid_instance",
            points=[
                (0, 0),
                (2, 3),
                (5, 4)
            ],
            demands=[
                0,
                2
            ],
            num_vehicles=2,
            vehicle_capacities=[
                10,
                10
            ]
        )


def test_vehicle_capacity_count_must_match():
    with pytest.raises(
        ValueError,
        match="A capacity must be provided"
    ):
        CVRPInstance(
            name="invalid_instance",
            points=[
                (0, 0),
                (2, 3)
            ],
            demands=[
                0,
                2
            ],
            num_vehicles=2,
            vehicle_capacities=[
                10
            ]
        )


def test_depot_demand_must_be_zero():
    with pytest.raises(
        ValueError,
        match="Depot demand must be zero"
    ):
        CVRPInstance(
            name="invalid_instance",
            points=[
                (0, 0),
                (2, 3)
            ],
            demands=[
                5,
                2
            ],
            num_vehicles=1,
            vehicle_capacities=[
                10
            ]
        )

def test_default_instance():
    instance = create_default_instance()

    assert instance.num_customers == 5
    assert instance.num_vehicles == 2
    assert instance.vehicle_capacities == [10, 10]
    assert instance.depot_index == 0
    