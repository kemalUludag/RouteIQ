from src.problem_data import depot, customers
from src.distance import calculate_distance


print(depot)
print(customers)


distance = calculate_distance(depot, customers[0])

print(distance)
