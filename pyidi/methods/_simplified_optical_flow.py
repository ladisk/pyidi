import numpy as np

from scipy.signal import convolve2d
from tqdm import tqdm
try:
    from qtpy.QtWidgets import QApplication
except ImportError:
    QApplication = None

import warnings
warnings.simplefilter("default")

from .idi_method import IDIMethod
from ._simplified_optical_flow_core import _compute_displacement_along_gradient, _compute_displacement_along_image_axis


class SimplifiedOpticalFlow(IDIMethod):
    """
    Displacmenet computation based on Simplified Optical Flow method [1].

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
    def configure(self, subset_size=3, pixel_shift=False, convert_from_px=1.,
        frame_range='all', mean_n_neighbours=0, zero_shift=False,
        progress_bar=True, reference_range=(0, 100)):
        """
        Set the attributes, compute reference image and gradients.
        
        :param video: 'parent' object
        :type video: object
        :param subset_size: size of the averaging subset, defaults to 3
        :param subset_size: int, optional
        :param pixel_shift: use pixel shift or not?, defaults to False
        :param pixel_shift: bool, optional
        :param convert_from_px: distance unit per pixel, defaults to 1.
        :param convert_from_px: float or int, optional
        :param frame_range: what range of images to calculate into displacements, defaults to 'all'
        :param frame_range: str or tuple, optional
        :param mean_n_neighbours: average the displacements of neighbouring points (how many points), defaults to 0
        :param mean_n_neighbours: int, optional
        :param zero_shift: shift the mean of the signal to zero?, defaults to False
        :param zero_shift: bool, optional
        :param progress_bar: show progress bar while calculating the displacements, defaults to True
        :param progress_bar: bool, optional
        :param reference_range: what range of images is averaged into reference image, defaults to (0, 100)
        :param reference_range: tuple, optional
        """
        if subset_size is not None:
            self.subset_size = subset_size
        if pixel_shift is not None:
            self.pixel_shift = pixel_shift
        if convert_from_px is not None:
            self.convert_from_px = convert_from_px
        if frame_range is not None:
            self.frame_range = frame_range
        if mean_n_neighbours is not None:
            self.mean_n_neighbours = mean_n_neighbours
        if zero_shift is not None:
            self.zero_shift = zero_shift
        if progress_bar is not None:
            self.progress_bar = progress_bar
        if reference_range is not None:
            self.reference_range = reference_range

        # Get reference image and gradients
        self.reference_image, self.gradient_0, self.gradient_1, self.gradient_magnitude = self._reference()

    def calculate_displacements(self):
        if not hasattr(self, 'points'):
            raise Exception('Please set points for analysis!')

        if self.pixel_shift:
            self.delta_0    = np.zeros((self.points.shape[0],)).astype(int)
            self.delta_1    = np.zeros((self.points.shape[0],)).astype(int)
            self.valid_points = np.ones((self.points.shape[0],)).astype(bool)
        else:
            self.delta_0 = 0
            self.delta_1 = 0

        gradient_0_direction = np.copy(self.gradient_0)
        gradient_1_direction = np.copy(self.gradient_1)

        signs_0 = np.sign(
            gradient_0_direction[self.points[:, 0], self.points[:, 1]])
        signs_1 = np.sign(
            gradient_1_direction[self.points[:, 0], self.points[:, 1]])

        self.direction_correction_0 = np.abs(
            gradient_0_direction[self.points[:, 0], self.points[:, 1]] / self.gradient_magnitude[self.points[:, 0], self.points[:, 1]])
        self.direction_correction_1 = np.abs(
            gradient_1_direction[self.points[:, 0], self.points[:, 1]] / self.gradient_magnitude[self.points[:, 0], self.points[:, 1]])

        # limited range of mraw can be observed
        if self.frame_range != 'all':
            limited_mraw = range(self.frame_range[0], self.frame_range[1])
            self.displacements  = np.zeros((self.points.shape[0], self.frame_range[1]-self.frame_range[0], 2))
        else:
            limited_mraw = range(self.video.N)
            self.displacements  = np.zeros((self.points.shape[0], self.video.N, 2))

        # Progress bar
        if self.progress_bar:
            p_bar = tqdm
        else:
            def p_bar(x, **kwargs): return x  # empty function

        # calculating the displacements
        for i, frame_number in enumerate(p_bar(limited_mraw, ncols=100)):
            image = self.video.get_frame(frame_number)
            image_filtered = self.subset(image, self.subset_size)

            self.image_roi = image_filtered[self.points[:,0] + self.delta_0, self.points[:, 1] + self.delta_1]
            self.latest_displacements = _compute_displacement_along_gradient(self.reference_image[self.points[:,0] , self.points[:, 1] ],
                                                                             self.image_roi, 
                                                                             self.gradient_magnitude[self.points[:,0], self.points[:, 1]])

            self.displacements[:, i] = _compute_displacement_along_image_axis(self.latest_displacements, signs_0, signs_1,
                                                                              self.direction_correction_0, self.direction_correction_1)
            self.displacements[:, i, 0] += self.delta_0
            self.displacements[:, i, 1] += self.delta_1

            if self.pixel_shift:
                self.pixel_shift_fun(i, self.points, image.shape)
            # Update the progress bar in the GUI
            if QApplication is not None and QApplication.instance() is not None:
                QApplication.processEvents()

        # Convert the displacements from pixels to physical units:
        self.displacements *= self.convert_from_px

        # average the neighbouring points
        if isinstance(self.mean_n_neighbours, int):
            if self.mean_n_neighbours > 0:
                self.displacement_averaging()

        # shift the mean of the signal to zero
        if isinstance(self.zero_shift, bool):
            if self.zero_shift is True:
                m = np.mean(self.displacements, axis=1)
                self.displacements[:, :, 0] -= m[:, 0:1]
                self.displacements[:, :, 1] -= m[:, 1:]

        #check for large displacements
        if np.max(self.displacements)/self.convert_from_px>0.5 and self.pixel_shift is False:
            warnings.warn('Displacement larger than 0.5 pixel')

    def displacement_averaging(self):
        """Calculate the average of displacements.
        """
        print('Averaging...')
        kernel = np.ones((self.mean_n_neighbours, 1)) / self.mean_n_neighbours

        d_0 = convolve2d(self.displacements[:, :, 0], kernel, mode='valid')[
            ::self.mean_n_neighbours]
        d_1 = convolve2d(self.displacements[:, :, 1], kernel, mode='valid')[
            ::self.mean_n_neighbours]

        self.displacements = np.concatenate(
            (d_0[:, :, np.newaxis], d_1[:, :, np.newaxis]), axis=2)
        print('Finished!')

    def pixel_shift_fun(self, i, points, image_shape):
        """Pixel shifting implementation. Points that are going outside of the image range are excluded.
        """
        self.delta_0 = np.round(self.displacements[:, i, 0]).astype(int)
        self.delta_1 = np.round(self.displacements[:, i, 1]).astype(int)
        
        # Exlude the points that have displacement going outside of the image range
        out_of_range_it = np.logical_or(self.delta_0 + points[:, 0] > image_shape[0] - 1, self.delta_1 + points[:, 1] > image_shape[1] - 1)
        if np.any(out_of_range_it):
            self.delta_0[out_of_range_it] = 0
            self.delta_1[out_of_range_it] = 0
            self.valid_points[out_of_range_it] = False
            warnings.warn('Displacement is going outside of the image range! The valid points are saved in self.method.valid_points')
        self.displacements[~self.valid_points, i, :] = np.nan

    def _reference(self):
        """Calculation of the reference image, image gradients and gradient amplitudes.

        :param images: Images to average. Usually the first 100 images.
        :param subset_size: Size of the subset to average.
        :return: Reference image, image gradient in 0 direction, image gradient in 1 direction, gradient magnitude
        """
        if self.reference_range[1] >= self.video.N:
            self.reference_range = (0, self.video.N)

        reference_image = np.zeros((self.video.image_height, \
                                    self.video.image_width), dtype=float)
        for frame in range(self.reference_range[0], self.reference_range[1]):
            reference_image += self.video.get_frame(frame)
        reference_image /= (self.reference_range[1] - self.reference_range[0])
        reference_image = self.subset(reference_image, self.subset_size)

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
