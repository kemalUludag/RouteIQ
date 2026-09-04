import random
from src.models import CVRPInstance

def generate_cvrp_instance(
    num_customers,
    num_vehicles,
    vehicle_capacity,
    seed=42
):
    random.seed(seed)

    depot = (0, 0)

    customers = [
        (
            random.randint(0, 20),
            random.randint(0, 20)
        )
        for _ in range(num_customers)
    ]

    while True:
        customer_demands = [
            random.randint(1, 4)
            for _ in range(num_customers)
        ]

        total_demand = sum(customer_demands)
        total_capacity = num_vehicles * vehicle_capacity

        if total_demand <= total_capacity:
            break

    demands = [0] + customer_demands

    return depot, customers, demands



def generate_cvrp_instance_model(
    num_customers,
    num_vehicles,
    vehicle_capacity,
    seed=42
):
    depot, customers, demands = (
        generate_cvrp_instance(
            num_customers=num_customers,
            num_vehicles=num_vehicles,
            vehicle_capacity=vehicle_capacity,
            seed=seed
        )
    )

    points = [depot] + customers

    vehicle_capacities = [
        vehicle_capacity
        for _ in range(num_vehicles)
    ]

    return CVRPInstance(
        name=f"cvrp_n{num_customers}_seed{seed}",
        points=points,
        demands=demands,
        num_vehicles=num_vehicles,
        vehicle_capacities=vehicle_capacities,
        depot_index=0
    )

