import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm_notebook as tqdm
from IPython.display import clear_output

from .idc_methods import *


class SimplifiedOpticalFlow(IDCMethods):
    def __init__(self, video, **kwargs):
        """
        :param video: 'parent' object
        :param kwargs: - subset_size
                       - pixel_shift
                       - contert_from_px
        """
        options = {
            'subset_size': 3,
            'pixel_shift': False,
            'convert_from_px': 1,
        }
        options.update(kwargs)

        self.subset_size = options['subset_size']
        self.pixel_shift = options['pixel_shift']
        self.convert_from_px = options['convert_from_px']

        self.reference_image, self.gradient_0, self.gradient_1, self.gradient_magnitude = self.reference(
            video.mraw[:100], self.subset_size)

        self.indices = video.points

    def calculate_displacements(self, video):
        self.displacements = np.zeros((video.N, self.indices.shape[0]))
        latest_displacements = 0
        signs = np.sign(
            self.gradient_0[self.indices[:, 0], self.indices[:, 1]])

        direction_correction = np.abs(
            self.gradient_0[self.indices[:, 0], self.indices[:, 1]] / self.gradient_magnitude[self.indices[:, 0], self.indices[:, 1]])

        for i, image in enumerate(tqdm(video.mraw[:5000])):
            image_filtered = self.subset(image, self.subset_size)
            
            image_roi = image_filtered[self.indices[:, 0], self.indices[:, 1]]
            
            latest_displacements = signs * \
                (self.reference_image[self.indices[:, 0], self.indices[:, 1]] - image_roi) / \
                self.gradient_magnitude[self.indices[:, 0], self.indices[:, 1]]
            
            self.displacements[i, :] = direction_correction * latest_displacements * self.convert_from_px

    def pixel_shift(self):
        """Pixel shifting implementation
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

    @staticmethod
    def show_points(video):
        """Showing the observed pixels on the image.
        """
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.imshow(video.mraw[0].astype(float), cmap='gray')
        ax.scatter(video.points[:, 1], video.points[:, 0], marker='.')
        if hasattr(video, 'polygon'):
            ax.add_patch(patches.Polygon(video.polygon, closed=True, color='w', fill=False, lw=1))
        plt.grid(False)
        plt.show()


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
        self.polygon = [[],[]]
        line, = ax.plot(self.polygon[0], self.polygon[1], 'r.-')
        def onclick(event):
            if event.button == 2: #če smo pritisnili gumb 2 (srednji na miški)
                if event.xdata is not None and event.ydata is not None:
                    self.polygon[0].append(int(np.round(event.xdata)))
                    self.polygon[1].append(int(np.round(event.ydata)))
                    print(f'x: {np.round(event.xdata):5.0f}, y: {np.round(event.ydata):5.0f}')
            elif event.button == 3:
                print('Deleted the last point...')
                del self.polygon[0][-1]
                del self.polygon[1][-1]

            line.set_xdata(self.polygon[0])
            line.set_ydata(self.polygon[1])
            fig.canvas.draw()

        def handle_close(event):
            """On closing."""
            try:
                clear_output() # pobrišemo prejšnje printe iz trenutne celice
            except:
                pass
            self.polygon = np.asarray(self.polygon).T
            for i, point in enumerate(self.polygon):
                print(f'{i+1}. point: x ={point[0]:5.0f}, y ={point[1]:5.0f}')
            
            # Add points to video object
            video.points = self.observed_pixels(gradient_0, n=n, points=self.polygon)
            video.polygon = self.polygon
            
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('close_event', handle_close) #ko zapremo figuro

    def observed_pixels(self, gradient_0, n, points):
        """Determine the observed pixels.

        Chooses the pixels with the highest gradient in 0 direction. Makes it suitable for beam like structures.
        Can be used to choose multiple points on one image width.

        :param gradient_0: Gradient in 0 direction. Used to determine observed pixels.
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
                    inside.append([y_, x_]) # Change indices (height, width)
        inside = np.asarray(inside) # Indices of points in the polygon

        g_inside = np.zeros_like(g)
        g_inside[inside[:, 0], inside[:, 1]] = g[inside[:, 0], inside[:, 1]]

        # Pixels with heighest gradient in 0 direction.
        for i in range(x_low, x_high):
            max_grad_ind = np.argsort(np.abs(g_inside[y_low:y_high, i]))[-n:] # n highest gradients
            if any(self.inside_polygon(i, g_, points) is not True for g_ in max_grad_ind+y_low):
                continue

            indices.append([[g_, i] for g_ in max_grad_ind+y_low]) # y_low has to be compensated for

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
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside