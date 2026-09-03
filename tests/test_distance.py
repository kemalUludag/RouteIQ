from src.distance import calculate_distance


def test_distance_3_4_5():
    result = calculate_distance((0, 0), (3, 4))

    assert result == 5.0

def test_distance_same_point():
    result = calculate_distance((2, 3), (2, 3))

    assert result == 0.0   

    