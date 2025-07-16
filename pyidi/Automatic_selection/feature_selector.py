import numpy as np
from scipy.ndimage import generic_filter

from .filters import ShiTomasi
available_filter_shortcuts = {'ST': ShiTomasi}

class FeatureSelector():
    """ Selects features from an image using different filters.
    eig0: The smallest eigenvalue of the structure tensor.
    harris: The Harris corner response function.
    trigs: The Triggs corner response function.
    harmonic_mean: The harmonic mean of the eigenvalues of the structure tensor.

    Args:
        image (ndarray): The image to select features from.
    """
    def __init__(self, image) -> None:
        self.image = image
        self.roi_size   = 9
        self.available_filter_shortcuts = available_filter_shortcuts
        return
        
    def set_filter(self, filter_shortcut):
        """
        Sets filter
        """
        if isinstance(filter_shortcut, str) and filter_shortcut in self.available_filter_shortcuts.keys():
            self.filter_shortcut = filter_shortcut
            self.filter = self.available_filter_shortcuts[filter_shortcut](self.image)
        else:
            "Filter shortcut not recognized"
    
    def set_roi(self, roi_size):
        self.roi_size = roi_size 

    #### Apply filter
    def apply_filter(self):
        row_of_interest = self.filter.n_layers//2
        if self.filter.parameter is None:
            score_image =  generic_filter(self.filter.to_filter, self.filter.filter, size=(self.roi_size, self.roi_size, self.filter.n_layers))[..., row_of_interest]
        else:
            score_image =  generic_filter(self.filter.to_filter, self.filter.filter, size=(self.roi_size, self.roi_size, self.filter.n_layers), extra_arguments = (self.filter.parameter, ))[..., row_of_interest]
        score_image[np.isnan(score_image)] = 0
        self.score_image = score_image
        return
    
    #### Pick points