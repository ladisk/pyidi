import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import scipy.signal

from .idi_method import IDIMethod


class GradientBasedOpticalFlow(IDIMethod):
    """Displacmenet computation based on Gradient Based Optical 
    Flow method [1] - a linearized, non-iterative approach.

    Literature:
        [1] Lucas, B. D., & Kanade, T. (1981). An Iterative Image Registration 
            Technique with an Application to Stereo Vision. In Proceedings of 
            the 7th International Joint Conference on Artificial 
            Intelligence - Volume 2 (pp. 674–679). San Francisco, CA, 
            USA: Morgan Kaufmann Publishers Inc.
    """

    def __init__(
        self, video, roi_size=(9, 9), kernel='central_fd',
        prefilter_gauss=True
    ):
        """Set attributes, set region of interest size.

        :param video: 'parent' object
        :type video: object
        :param roi_size: size of the region of interest, defaults to (9, 9)
        :param roi_size: tuple or int, optional
        :param kernel: kernel for convolution, defaults to 'central_fd'
        :param kernel: str, optional
        :param prefilter_gauss: use Gauss filter, defaults to True
        :param prefilter_gauss: bool, optional
        """

        self.roi_size = roi_size
        self.kernel = kernel
        self.prefilter_gauss = prefilter_gauss

        self._set_roi_size(self.roi_size)


    def calculate_displacements(self, video, roi_size=None):
        """Get the displacements of all selected points/roi_references.

        Calls a `get_simple_translation` method that calculates a displacement
        for one point only.

        :param video: parent object
        :type video: object
        """
        if roi_size is not None:
            self._set_roi_size(roi_size)

        self.displacements = []
        for roi_ref in tqdm(video.points):
            single_roi_translation = self.get_simple_translation(
                video, roi_ref)
            self.displacements.append(single_roi_translation)
        self.displacements = np.asarray(self.displacements)


    def get_simple_translation(self, video, roi_reference):
        """Onle point/roi_reference caluclation.

        Calculates a displacement for one point/roi_reference only.

        :param video: parent object
        :type video: object
        :param roi_reference: Center vertex of RIO
        :type roi_reference: tuple
        :return: Displacements in y and x direction
        :rtype: numpy array
        """
        roi_reference = np.asarray(roi_reference)
        # First ROI image, used for the initial guess.
        F = self._get_roi_image(video.mraw[0], roi_reference).astype(float)

        # Initialize the results array.
        results = np.array([[0, 0]], dtype=np.float64)
        # Initialize a reference for all following calculations.
        p_ref = roi_reference

        # First image was loaded already.
        for i in range(1, len(video.mraw)):
            # Last calculated integer translation.
            d_int = np.round(results[-1])
            # Current image at integer location.
            G = self._get_roi_image(video.mraw[i], p_ref + d_int).astype(float)
            Gx, Gy = self.get_gradient(G)

            Gx2 = np.sum(Gx**2)
            Gy2 = np.sum(Gy**2)
            GxGy = np.sum(Gx * Gy)

            A = np.array([[Gx2, GxGy],
                          [GxGy, Gy2]])

            b = np.array([np.sum(Gx*(F - G)), np.sum(Gy*(F-G))])

            d = np.linalg.solve(A, b)  # dx, dy

            results = np.vstack((results, d_int-d[::-1]))  # y, x
        return results


    def _get_roi_image(self, target, roi_reference):
        '''Get 2D ROI array from target image, ROI position and size.

        :param target: Target iamge.
        :param roi_reference: Center coordinate point of ROI, (y, x).
        :return: ROI image (2D numpy array).
        '''
        ul = (np.array(roi_reference) - np.array(self.roi_size) //
              2).astype(int)  # Center vertex of ROI
        m, n = target.shape
        ul = np.clip(np.array(ul), 0, [
                     m-self.roi_size[0]-1, n-self.roi_size[1]-1])
        roi_image = target[ul[0]:ul[0]+self.roi_size[0],
                           ul[1]:ul[1]+self.roi_size[1]]
        return roi_image


    def get_gradient(self, image):
        '''Computes gradient of inputimage, using the specified convoluton kernels.

        :param image: Image to compute gradient of.
        :return: [gx, gy] (numpy array): Gradient images with respect to x and y direction.
        '''
        if self.kernel == 'central_fd':
            if self.prefilter_gauss:
                # x_kernel = np.array([[-0.14086616, -0.20863973,  0.,  0.20863973,  0.14086616]])
                x_kernel = np.array([[-0.44637882,  0.,  0.44637882]])
            else:
                # x_kernel = np.array([[1, -8, 0, 8, -1]], dtype=float)/12
                x_kernel = np.array([[-0.5,  0.,  0.5]])
            y_kernel = np.transpose(x_kernel)
        elif not isinstance(self.kernel, str) and len(self.kernel) == 2 and self.kernel[0].shape[1] >= 3 and self.kernel[1].shape[0] >= 3:
            x_kernel = self.kernel[0]
            y_kernel = self.kernel[1]
        else:
            raise ValueError(
                'Please input valid gradient convolution kernels!')

        g_x = scipy.signal.convolve2d(
            image.astype(float), x_kernel, mode='same')
        g_y = scipy.signal.convolve2d(
            image.astype(float), y_kernel, mode='same')
        return np.array([g_x, g_y], dtype=np.float64)


    def show_points(self, video, roi_size=None):
        """Show points to be analyzed, together with ROI borders.
        """
        if roi_size is None:
            if hasattr(self, 'roi_size'):
                roi_size = self.roi_size

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.imshow(video.mraw[0].astype(float), cmap='gray')
        ax.scatter(video.points[:, 1],
                   video.points[:, 0], marker='.', color='r')

        if roi_size is not None:
            for point in video.points:
                roi_border = patches.Rectangle((point - self.roi_size//2)[::-1], self.roi_size[1], self.roi_size[0],
                                               linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(roi_border)

        plt.grid(False)
        plt.show()


    def _set_roi_size(self, roi_size):
        """Set ROI size for displacement identification.
        """
        if isinstance(roi_size, int):
            self.roi_size = np.array([roi_size, roi_size], dtype=int)
        else:
            self.roi_size = np.array(roi_size, dtype=int)
        if np.sum(self.roi_size) < 40:
            print(
                'WARNING: Selected region of interest is small. For better results select larger ROI size.')

    @staticmethod
    def get_points(video, **kwargs):
        print('Point/ROI selection is not yet implemented!')
        pass
