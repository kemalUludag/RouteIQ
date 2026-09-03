import random


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
