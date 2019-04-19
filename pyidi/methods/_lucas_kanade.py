import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

import scipy.signal
from scipy.interpolate import interp2d
import scipy.optimize
from tqdm import tqdm as tqdm
from tqdm import trange

from .idi_method import IDIMethod

class LucasKanade(IDIMethod):

    def __init__(
        self, video, roi_size=3, pad=2, max_nfev=20, verbose=1, show_pbar=True
    ):
        self.pad = pad
        self.max_nfev = max_nfev
        self.verbose = verbose
        self.show_pbar = show_pbar
        self.roi_size = roi_size
        self._set_roi_size(self.roi_size)


    def calculate_displacements(self, video, roi_size=None):
        
        def opt(d, p, G):
            """Optimization function.
            """
            F_current = self.F_int[p](self.extended_points_0[p, self.pad:-self.pad] - d[0], self.extended_points_1[p, self.pad:-self.pad] - d[1])
            return (F_current - G).flatten()
        
        if roi_size is not None:
            self._set_roi_size(roi_size)

        self.displacements = np.zeros((video.points.shape[0], video.N, 2))

        start_time = time.time()

        extend_0 = np.arange(0, self.roi_size[0] + 2*self.pad) - self.roi_size[0]//2 - self.pad
        extend_1 = np.arange(0, self.roi_size[1] + 2*self.pad) - self.roi_size[1]//2 - self.pad

        self.extended_points_0 = np.vstack(video.points[:, 0]) + extend_0
        self.extended_points_1 = np.vstack(video.points[:, 1]) + extend_1

        if self.verbose:
            t = time.time()
            print(f'Interpolating the reference image...')
        self._interpolation(video)

        if self.verbose:
            print(f'...done in {time.time() - t:.2f} s')

        # Time iteration.
        for i in self._pbar(1, len(video.mraw)):
            # Iterate over points.
            for p in range(video.points.shape[0]):
                G = video.mraw[i, self.mgrid_0[p], self.mgrid_1[p]]
                
                delta = self.displacements[p, i-1] # Initial value
                sol = scipy.optimize.least_squares(lambda x: opt(x, p, G), delta, max_nfev=self.max_nfev) # Optimization

                self.displacements[p, i] = sol.x
        
        
        if self.verbose:
            full_time = time.time() - start_time
            if full_time > 60:
                full_time_m = full_time//60
                full_time_s = full_time%60
                print(f'Time to complete: {full_time_m:0f} min, {full_time_s:.1f} s')
            else:
                print(f'Time to complete: {full_time:1f} s')
    

    def _pbar(self, x, y):
        if self.show_pbar:
            return trange(x, y, ncols=100, leave=True)
        else:
            return range(x, y)


    def _interpolation(self, video):
        self.F_int = []
        self.mgrid_0 = []
        self.mgrid_1 = []
        for p in range(video.points.shape[0]):
            _m_0, _m_1 = np.meshgrid(self.extended_points_0[p], self.extended_points_1[p])
            _F_int = interp2d(self.extended_points_0[p], self.extended_points_1[p], video.mraw[0, _m_0, _m_1], kind='cubic')
            self.F_int.append(_F_int)

            m_0, m_1 = np.meshgrid(self.extended_points_0[p, self.pad:-self.pad], self.extended_points_1[p, self.pad:-self.pad])
            self.mgrid_0.append(m_0)
            self.mgrid_1.append(m_1)

    # def calculate_displacements(self, video, roi_size=None):
    #     if roi_size is not None:
    #         self._set_roi_size(roi_size)

    #     self.displacements = np.zeros((video.points.shape[0], video.N, 2))
    #     start_time = time.time()
    #     for p in range(len(video.points)):
    #         self.tqdm_text = f'{p+1}/{len(video.points)}'
    #         single_roi_translation = self.get_simple_translation(video, video.points[p])
    #         self.displacements[p] = single_roi_translation

    #     full_time = time.time() - start_time
    #     if full_time > 60:
    #         full_time_m = full_time//60
    #         full_time_s = full_time%60
    #         print(f'Time to complete: {full_time_m:0f} min, {full_time_s:.1f} s')
    #     else:
    #         print(f'Time to complete: {full_time:1f} s')


    # def get_simple_translation(self, video, roi_reference):
    #     """Onle point/roi_reference caluclation.

    #     Calculates a displacement for one point/roi_reference only.

    #     :param video: parent object
    #     :type video: object
    #     :param roi_reference: Center vertex of RIO
    #     :type roi_reference: tuple
    #     :return: Displacements in y and x direction
    #     :rtype: numpy array
    #     """
    #     roi_reference = np.asarray(roi_reference)
        
    #     pad = 4
    #     pad2 = int(pad/2)
    #     roi_coors_0 = np.arange(0, self.roi_size[0]+pad) - self.roi_size[0]//2 - pad2
    #     roi_coors_1 = np.arange(0, self.roi_size[1]+pad) - self.roi_size[1]//2 - pad2

    #     F = self._get_roi_image(video.mraw[0], roi_reference, self.roi_size + pad).astype(float)
    #     F_int = interp2d(roi_coors_1, roi_coors_0, F, kind='cubic')

    #     # Initialize the results array.
    #     results = np.array([[0, 0]], dtype=np.float64)
        
    #     for i in trange(1, len(video.mraw), ncols=100, desc=self.tqdm_text, leave=True):
    #         G = self._get_roi_image(video.mraw[i], roi_reference, self.roi_size).astype(float)
            
    #         delta = np.copy(results[-1])
    #         def opt(d):
    #             F_current = F_int(roi_coors_1[pad2:-pad2] - d[1], roi_coors_0[pad2:-pad2] - d[0])
    #             return (F_current - G).flatten()
            
    #         sol = scipy.optimize.least_squares(opt, delta, max_nfev=10)

    #         results = np.vstack((results, sol.x))

    #     return results
            

    # def _get_roi_image(self, target, roi_reference, roi_size):
    #     '''Get 2D ROI array from target image, ROI position and size.

    #     :param target: Target iamge.
    #     :param roi_reference: Center coordinate point of ROI, (y, x).
    #     :return: ROI image (2D numpy array).
    #     '''
    #     ul = (np.array(roi_reference) - np.array(roi_size) //
    #           2).astype(int)  # Center vertex of ROI
    #     m, n = target.shape
    #     ul = np.clip(np.array(ul), 0, [
    #                  m-roi_size[0]-1, n-roi_size[1]-1])
    #     roi_image = target[ul[0]:ul[0]+roi_size[0],
    #                        ul[1]:ul[1]+roi_size[1]]
    #     return roi_image


    # def get_gradient(self, image):
    #     '''Computes gradient of inputimage, using the specified convoluton kernels.

    #     :param image: Image to compute gradient of.
    #     :return: [gx, gy] (numpy array): Gradient images with respect to x and y direction.
    #     '''
    #     if self.kernel == 'central_fd':
    #         if self.prefilter_gauss:
    #             # x_kernel = np.array([[-0.14086616, -0.20863973,  0.,  0.20863973,  0.14086616]])
    #             x_kernel = np.array([[-0.44637882,  0.,  0.44637882]])
    #         else:
    #             # x_kernel = np.array([[1, -8, 0, 8, -1]], dtype=float)/12
    #             x_kernel = np.array([[-0.5,  0.,  0.5]])
    #         y_kernel = np.transpose(x_kernel)
    #     elif not isinstance(self.kernel, str) and len(self.kernel) == 2 and self.kernel[0].shape[1] >= 3 and self.kernel[1].shape[0] >= 3:
    #         x_kernel = self.kernel[0]
    #         y_kernel = self.kernel[1]
    #     else:
    #         raise ValueError(
    #             'Please input valid gradient convolution kernels!')

    #     g_x = scipy.signal.convolve2d(
    #         image.astype(float), x_kernel, mode='same')
    #     g_y = scipy.signal.convolve2d(
    #         image.astype(float), y_kernel, mode='same')
    #     return np.array([g_x, g_y], dtype=np.float64)


    def _set_roi_size(self, roi_size):
        """Set ROI size for displacement identification.
        """
        if isinstance(roi_size, int):
            self.roi_size = np.array([roi_size, roi_size], dtype=int)
        else:
            self.roi_size = np.array(roi_size, dtype=int)


    @staticmethod
    def get_points():
        pass