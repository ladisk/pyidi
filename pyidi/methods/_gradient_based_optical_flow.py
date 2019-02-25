import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import scipy.signal

from .idi_method import IDIMethod

class GradientBasedOpticalFlow(IDIMethod):

    def __init__(self, video, **kwargs):
        """
        :param video: 'parent' object
        :param kwargs: keyword arguments (defined in `options`)
        """
        options = {
            'roi_size': (9, 9),
            'kernel': 'central_fd', # Tuple of convolution kernels in x and y direction. Central finite difference used if left blank.
            'prefilter_gauss': True, # If True, the gradient kernel is first filtered with a Gauss filter to eliminate noise.
        }

        # Check for valid kwargs
        for kwarg in kwargs.keys():
            if kwarg not in options.keys():
                raise Exception(f'keyword argument "{kwarg}" is not one of the options for this method')

        options.update(kwargs) # Update the options dict
        self.__dict__.update(options) # Update the objects attributes

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
            single_roi_translation = self.get_simple_translation(video, roi_ref)
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
        F = self._get_roi_image(video.mraw[0], roi_reference).astype(float)  # First ROI image, used for the initial guess.

        results = np.array([[0, 0]], dtype=np.float64)          # Initialize the results array.
        p_ref = roi_reference                                   # Initialize a reference for all following calculations.

        for i in range(1, len(video.mraw)):                         # First image was loaded already.
            d_int = np.round(results[-1])                       # Last calculated integer translation.
            G = self._get_roi_image(video.mraw[i], p_ref + d_int).astype(float) # Current image at integer location.
            Gx, Gy = self.get_gradient(G)

            Gx2 = np.sum(Gx**2)
            Gy2 = np.sum(Gy**2)
            GxGy = np.sum(Gx * Gy)

            A = np.array([[Gx2, GxGy],
                        [GxGy, Gy2]])
            
            b = np.array([np.sum(Gx*(F - G)), np.sum(Gy*(F-G))])

            d = np.linalg.solve(A, b) # dx, dy

            results = np.vstack((results, d_int-d[::-1])) # y, x
        return results

    
    def _get_roi_image(self, target, roi_reference):
        '''Get 2D ROI array from target image, ROI position and size.
        
        :param target: Target iamge.
        :param roi_reference: Center coordinate point of ROI, (y, x).
        :return: ROI image (2D numpy array).
        '''
        ul = (np.array(roi_reference) - np.array(self.roi_size)//2).astype(int)  # Center vertex of ROI
        m, n = target.shape
        ul = np.clip(np.array(ul), 0, [m-self.roi_size[0]-1, n-self.roi_size[1]-1])
        roi_image = target[ul[0]:ul[0]+self.roi_size[0], ul[1]:ul[1]+self.roi_size[1]]
        return roi_image


    def get_gradient(self, image):
        '''Computes gradient of inputimage, using the specified convoluton kernels.
        
        :param image: Image to compute gradient of.
        :return: [gx, gy] (numpy array): Gradient images with respect to x and y direction.
        '''
        if self.kernel == 'central_fd':
            if self.prefilter_gauss:
                # x_kernel = np.array([[-0.14086616, -0.20863973,  0.,  0.20863973,  0.14086616]])
                x_kernel = np.array([[-0.44637882,  0.        ,  0.44637882]])
            else:
                # x_kernel = np.array([[1, -8, 0, 8, -1]], dtype=float)/12
                x_kernel = np.array([[-0.5,  0. ,  0.5]])
            y_kernel = np.transpose(x_kernel)
        elif not isinstance(self.kernel, str) and len(self.kernel) == 2 and self.kernel[0].shape[1] >= 3 and self.kernel[1].shape[0] >= 3:
            x_kernel = self.kernel[0]
            y_kernel = self.kernel[1]
        else:
            raise ValueError('Please input valid gradient convolution kernels!')

        g_x = scipy.signal.convolve2d(image.astype(float), x_kernel, mode='same')
        g_y = scipy.signal.convolve2d(image.astype(float), y_kernel, mode='same')
        return np.array([g_x, g_y], dtype=np.float64)

    def show_points(self, video, roi_size=None):
        """
        Show points to be analyzed, together with ROI borders.
        """

        if roi_size is None:
            if hasattr(self, 'roi_size'):
                roi_size = self.roi_size

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.imshow(video.mraw[0].astype(float), cmap='gray')
        ax.scatter(video.points[:, 1], video.points[:, 0], marker='.', color='r')

        if roi_size is not None:
            for point in video.points:
                roi_border = patches.Rectangle((point - self.roi_size//2)[::-1], self.roi_size[1], self.roi_size[0], 
                    linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(roi_border)

        plt.grid(False)
        plt.show()


    def _set_roi_size(self, roi_size):
        """
        Set ROI size for displacement identification.
        """
        if isinstance(roi_size, int):
            self.roi_size = np.array([roi_size, roi_size], dtype=int)
        else:
            self.roi_size = np.array(roi_size, dtype=int)
        if np.sum(self.roi_size) < 40:
            print('WARNING: Selected region of interest is small. For better results select larger ROI size.')


    @staticmethod
    def get_points(video, **kwargs):
        print('Point/ROI selection is not yet implemented!')
        pass

    