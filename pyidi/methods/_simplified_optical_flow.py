import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

from .idi_method import IDIMethod


class SimplifiedOpticalFlow(IDIMethod):
    """Displacmenet computation based on Simplified Optical Flow method [1].

    Literature:
        [1] Javh, J., Slavič, J., & Boltežar, M. (2017). The subpixel resolution 
            of optical-flow-based modal analysis. Mechanical Systems 
            and Signal Processing, 88, 89–99.
        [2] Lucas, B. D., & Kanade, T. (1981). An Iterative Image Registration 
            Technique with an Application to Stereo Vision. In Proceedings of 
            the 7th International Joint Conference on Artificial 
            Intelligence - Volume 2 (pp. 674–679). San Francisco, CA, 
            USA: Morgan Kaufmann Publishers Inc.
    """

    def __init__(self, video, **kwargs):
        """
        :param video: 'parent' object
        :param kwargs: keyword arguments (defined in `options`)
        """
        options = {
            'subset_size': 3,
            'pixel_shift': False,
            'convert_from_px': 1,
            'mraw_range': 'all',
            'mean_n_neighbours': 0,
            'zero_shift': False,
            'progress_bar': True,
        }
        options.update(kwargs)

        self.subset_size = options['subset_size']
        self.pixel_shift = options['pixel_shift']
        self.convert_from_px = options['convert_from_px']
        self.mraw_range = options['mraw_range']
        self.mean_n_neighbours = options['mean_n_neighbours']
        self.zero_shift = options['zero_shift']
        self.progress_bar = options['progress_bar']

        self.reference_image, self.gradient_0, self.gradient_1, self.gradient_magnitude = self.reference(
            video.mraw[:100], self.subset_size)

        self.indices = video.points

    def calculate_displacements(self, video):
        self.displacements = np.zeros((self.indices.shape[0], video.N, 2))
        latest_displacements = 0

        gradient_0_direction = np.copy(self.gradient_0)
        gradient_1_direction = np.copy(self.gradient_1)

        signs_0 = np.sign(
            gradient_0_direction[self.indices[:, 0], self.indices[:, 1]])
        signs_1 = np.sign(
            gradient_1_direction[self.indices[:, 0], self.indices[:, 1]])

        self.direction_correction_0 = np.abs(
            gradient_0_direction[self.indices[:, 0], self.indices[:, 1]] / self.gradient_magnitude[self.indices[:, 0], self.indices[:, 1]])
        self.direction_correction_1 = np.abs(
            gradient_1_direction[self.indices[:, 0], self.indices[:, 1]] / self.gradient_magnitude[self.indices[:, 0], self.indices[:, 1]])

        # limited range of mraw can be observed
        if self.mraw_range != 'all':
            limited_mraw = video.mraw[self.mraw_range[0]: self.mraw_range[1]]
        else:
            limited_mraw = video.mraw

        # Progress bar
        if self.progress_bar:
            p_bar = tqdm
        else:
            p_bar = lambda x: x # empty function

        # calculating the displacements
        for i, image in enumerate(p_bar(limited_mraw)):
            image_filtered = self.subset(image, self.subset_size)

            if self.pixel_shift:
                print('Pixel-shifting is not yet implemented.')
                break

            else:
                self.image_roi = image_filtered[self.indices[:, 0], self.indices[:, 1]]

                self.latest_displacements = (self.reference_image[self.indices[:, 0], self.indices[:, 1]] - self.image_roi) / \
                    self.gradient_magnitude[self.indices[:, 0], self.indices[:, 1]]

            self.displacements[:, i, 0] = signs_0 * self.direction_correction_0 * \
                self.latest_displacements * self.convert_from_px
            self.displacements[:, i, 1] = signs_1 * self.direction_correction_1 * \
                self.latest_displacements * self.convert_from_px

        # average the neighbouring points
        if isinstance(self.mean_n_neighbours, int):
            if self.mean_n_neighbours > 0:
                self.displacement_averaging()

        # shift the mean of the signal to zero
        if isinstance(self.zero_shift, bool):
            if self.zero_shift is True:
                self.displacements -= np.mean(self.displacements, axis=0)

    def displacement_averaging(self):
        """Calculate the average of displacements.
        """
        print('Averaging...')
        reshaped = self.displacements.reshape(
            self.displacements.shape[0],
            self.displacements.shape[1]//(self.mean_n_neighbours),
            self.mean_n_neighbours)

        self.displacements = np.mean(reshaped, axis=2)
        print('Finished!')

    def pixel_shift(self):
        """Pixel shifting implementation.
        """
        pass

    def reference(self, images, subset_size):
        """Calculation of the reference image, image gradients and gradient amplitudes.

        :param images: Images to average. Usually the first 100 images.
        :param subset_size: Size of the subset to average.
        :return: Reference image, image gradient in 0 direction, image gradient in 1 direction, gradient magnitude
        """
        reference_image = np.mean([self.subset(image_, subset_size)
                                   for image_ in images], axis=0)

        gradient_0, gradient_1 = np.gradient(reference_image)
        gradient_magnitude = np.sqrt(gradient_0**2 + gradient_1**2)

        return reference_image, gradient_0, gradient_1, gradient_magnitude

    def subset(self, data, subset_size):
        """Calculating a filtered image.

        Calculates a filtered image with subset of d. It sums the area of d x d.

        :param data: Image that is to be filtered.
        :param subset_size: Size of the subset.
        :return: Filtered image.
        """
        subset_size_q = int((subset_size - 1) / 2)
        subset_image = []

        for i in range(-subset_size_q, subset_size_q + 1):
            for j in range(-subset_size_q, subset_size_q + 1):
                subset_roll = np.roll(data, i, axis=0)
                subset_roll = np.roll(subset_roll, j, axis=1)
                subset_image.append(subset_roll)

        return np.sum(np.asarray(subset_image), axis=0)

    @staticmethod
    def get_points(video, **kwargs):
        """Determine the points.
        """
        options = {
            'n': 1,
        }

        options.update(kwargs)

        polygon = PickPoints(video, n=options['n'])


class PickPoints:
    """Pick the area of interest.

    Select the points with highest gradient in vertical direction.
    """

    def __init__(self, video, n):
        image = video.mraw[0]
        gradient_0, gradient_1 = np.gradient(image.astype(float))

        self.corners = []
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.grid(False)
        ax.imshow(image, cmap='gray')
        self.polygon = [[], []]
        line, = ax.plot(self.polygon[0], self.polygon[1], 'r.-')

        def onclick(event):
            if event.button == 2:
                if event.xdata is not None and event.ydata is not None:
                    self.polygon[0].append(int(np.round(event.xdata)))
                    self.polygon[1].append(int(np.round(event.ydata)))
                    print(
                        f'x: {np.round(event.xdata):5.0f}, y: {np.round(event.ydata):5.0f}')
            elif event.button == 3:
                print('Deleted the last point...')
                del self.polygon[0][-1]
                del self.polygon[1][-1]

            line.set_xdata(self.polygon[0])
            line.set_ydata(self.polygon[1])
            fig.canvas.draw()

        def handle_close(event):
            """On closing."""
            self.polygon = np.asarray(self.polygon).T
            for i, point in enumerate(self.polygon):
                print(f'{i+1}. point: x ={point[0]:5.0f}, y ={point[1]:5.0f}')

            # Add points to video object
            video.points = self.observed_pixels(
                gradient_0, gradient_1, n=n, points=self.polygon)
            video.polygon = self.polygon

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect(
            'close_event', handle_close)  # on closing the figure

    def observed_pixels(self, gradient_0, gradient_1, n, points):
        """Determine the observed pixels.

        Chooses the pixels with the highest gradient in 0 direction. Makes it suitable for beam like structures.
        Can be used to choose multiple points on one image width.

        :param n: Number of points on every image width. 
        :param points: Polygon points. Pixels are chosen only from within the polygon.
        :return: Indices of observed pixels
        """

        g = np.copy(gradient_0)
        indices = []
        inside = []
        x = points[:, 0]
        y = points[:, 1]

        x_low = min(x)
        x_high = max(x)
        y_low = min(y)
        y_high = max(y)
        for x_ in range(x_low, x_high):
            for y_ in range(y_low, y_high):
                if self.inside_polygon(x_, y_, points) is True:
                    inside.append([y_, x_])  # Change indices (height, width)
        inside = np.asarray(inside)  # Indices of points in the polygon

        g_inside = np.zeros_like(g)
        g_inside[inside[:, 0], inside[:, 1]] = g[inside[:, 0], inside[:, 1]]

        # Pixels with heighest gradient in 0 direction.
        for i in range(x_low, x_high):
            max_grad_ind = np.argsort(
                np.abs(g_inside[y_low:y_high, i]))[-n:]  # n highest gradients
            if any(self.inside_polygon(i, g_, points) is not True for g_ in max_grad_ind+y_low):
                continue

            # y_low has to be compensated for
            indices.append([[g_, i] for g_ in max_grad_ind+y_low])

        return np.asarray(indices).reshape(n*(x_low-x_high), 2)

    def inside_polygon(self, x, y, points):
        """Return True if a coordinate (x, y) is inside a polygon defined by
        a list of verticies [(x1, y1), (x2, x2), ... , (xN, yN)].

        Reference: http://www.ariel.com.au/a/python-point-int-poly.html
        """
        n = len(points)
        inside = False
        p1x, p1y = points[0]
        for i in range(1, n + 1):
            p2x, p2y = points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / \
                                (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
