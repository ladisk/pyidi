import numpy as np

def points_along_polygon(polygon, subset_size):
    if len(polygon) < 2:
        return []

    # List of points along the path
    result_points = []

    for i in range(len(polygon) - 1):
        p1 = np.array(polygon[i])
        p2 = np.array(polygon[i + 1])
        segment = p2 - p1
        length = np.linalg.norm(segment)

        if length == 0:
            continue

        direction = segment / length
        n_points = int(length // subset_size)

        for j in range(n_points + 1):
            pt = p1 + j * subset_size * direction
            result_points.append((round(pt[0] - 0.5) + 0.5, round(pt[1] - 0.5) + 0.5))

    return result_points