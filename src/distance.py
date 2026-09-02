import math
def calculate_distance(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def create_distance_matrix(points):
     
    num_points = len(points)
    distance_matrix = [[0] * num_points for _ in range(num_points)]

    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                distance_matrix[i][j] = calculate_distance(points[i], points[j])

    return distance_matrix
