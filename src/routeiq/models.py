from dataclasses import dataclass, field
from typing import Any


@dataclass
class CVRPInstance:
    name: str

    points: list[tuple[float, float]]
    demands: list[int]

    num_vehicles: int
    vehicle_capacities: list[int]

    depot_index: int = 0

    def __post_init__(self):
        if len(self.points) != len(self.demands):
            raise ValueError(
                "The number of points must match the number of demands."
            )

        if self.num_vehicles <= 0:
            raise ValueError(
                "The number of vehicles must be positive."
            )

        if len(self.vehicle_capacities) != self.num_vehicles:
            raise ValueError(
                "A capacity must be provided for every vehicle."
            )

        if any(capacity <= 0 for capacity in self.vehicle_capacities):
            raise ValueError(
                "Vehicle capacities must be positive."
            )

        if self.demands[self.depot_index] != 0:
            raise ValueError(
                "Depot demand must be zero."
            )

    @property
    def num_nodes(self):
        return len(self.points)

    @property
    def num_customers(self):
        return self.num_nodes - 1


@dataclass
class Solution:
    solver_name: str

    routes: list[list[int]]

    total_distance: float
    runtime_seconds: float

    feasible: bool
    status: str

    metadata: dict[str, Any] = field(
        default_factory=dict
    )
    