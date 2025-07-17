import numpy as np
from scipy.ndimage import maximum_filter
from skimage.feature import peak_local_max

class PickingBase:
    parameters = ['n_points']
    def __init__(self, n_points=None):
        self.n_points = n_points
        self.maxima = None
    
    def handle_too_few_points(self):
        if self.n_points is not None and self.maxima.shape[0] < self.n_points:
            print(f"More points requested than available: {self.maxima.shape[0]} points set.")
        return


class LocalMaxima(PickingBase):
    parameters = ['n_points', 'min_distance']

    def __init__(self, min_distance=5, n_points=None):
        super().__init__(n_points=n_points)
        self.min_distance = min_distance

    def pick(self, score_image):
        threshold = 0 if self.n_points is not None else np.max(score_image) * 0.9

        maxima = maximum_filter(score_image, size=self.min_distance)
        maxima = (score_image == maxima) & (score_image > threshold)
        self.maxima = np.argwhere(maxima)

        if self.n_points is not None and self.maxima.shape[0] > self.n_points:
            order = np.argsort(score_image[self.maxima[:, 0], self.maxima[:, 1]])[::-1]
            self.maxima = self.maxima[order[:self.n_points]]
        self.handle_too_few_points()
        return self.maxima

class ANMS(PickingBase):
    def __init__(self, n_points=100, **kwargs):
        super().__init__(n_points=n_points, **kwargs)
        self.radii = None

    def pick(self, score_image):
        # Use score image from FeatureSelector
        local_max_picker = LocalMaxima(min_distance=1)
        coords = local_max_picker.pick(score_image)
        scores = score_image[coords[:, 0], coords[:, 1]]

        if len(coords) <= self.n_points:
            self.maxima = coords
            self.radii = np.ones(len(coords))
            print(f"More points requested than available: {self.maxima.shape[0]} points set.")
            return self.maxima

        # Sort descending by score
        sorted_indices = np.argsort(-scores)
        sorted_coords = coords[sorted_indices]
        # sorted_scores = scores[sorted_indices]

        radii = np.full(len(sorted_coords), np.inf)

        for i in range(1, len(sorted_coords)):
            stronger = sorted_coords[:i]
            dists = np.linalg.norm(stronger - sorted_coords[i], axis=1)
            radii[i] = np.min(dists)

        top_indices = np.argsort(-radii)[:self.n_points]
        self.maxima = sorted_coords[top_indices]
        self.radii = radii[top_indices]
        return self.maxima

class DescendingScore(PickingBase):
    parameters = ['n_points', 'min_distance', 'min_score']

    def __init__(self, min_distance=5, min_score=0, n_points=None):
        super().__init__(n_points=n_points)
        self.min_distance = min_distance
        self.min_score = min_score

    def pick(self, score_image):
        if isinstance(self.min_distance, (int, float)):
            min_distance = (self.min_distance, self.min_distance)
        else:
            min_distance = self.min_distance

        qi, qj = (min_distance[0] - 1), (min_distance[1] - 1)
        si_flat = score_image.flatten()
        score_order = np.argsort(si_flat)[::-1]

        first_low_score = np.argmax(si_flat[score_order] < self.min_score)
        score_order = score_order[:first_low_score]

        placed_points = np.zeros_like(score_image, dtype=bool)
        maxima_list = []

        for point in score_order:
            y, x = np.unravel_index(point, score_image.shape)
            y_start, y_end = max(y - qi, 0), min(y + qi + 1, score_image.shape[0])
            x_start, x_end = max(x - qj, 0), min(x + qj + 1, score_image.shape[1])
            if placed_points[y_start:y_end, x_start:x_end].any():
                continue

            placed_points[y, x] = True
            maxima_list.append([y, x])

            if len(maxima_list) >= self.n_points:
                break

        self.maxima = np.array(maxima_list)
        self.handle_too_few_points()
        return self.maxima