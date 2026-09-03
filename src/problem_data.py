from src.models import CVRPInstance

depot = (0, 0)

customers = [
    (2, 3),
    (5, 4),
    (1, 7),
    (6, 8),
    (8, 2)
]
demands = [0, 2, 4, 3, 5, 2]

num_vehicles = 2
vehicle_capacity = 10

def create_default_instance():
    points = [depot] + customers

    vehicle_capacities = [
        vehicle_capacity
        for _ in range(num_vehicles)
    ]

    return CVRPInstance(
        name="default_cvrp_instance",
        points=points,
        demands=demands,
        num_vehicles=num_vehicles,
        vehicle_capacities=vehicle_capacities,
        depot_index=0
    )
