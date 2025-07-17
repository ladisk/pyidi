import numpy as np
from scipy.ndimage import generic_filter

from .filters import ShiTomasi, DirectionalFilter, Harris, Triggs, HarmonicMean

available_filter_shortcuts = {
    'ST': ShiTomasi,
    'DF': DirectionalFilter,
    'HARRIS': Harris,
    'TRIGGS': Triggs,
    'HM': HarmonicMean,
}
from .picking_methods import LocalMaxima, ANMS, DescendingScore
available_pickers_shortcuts = {
    'LM': LocalMaxima,
    'ANMS': ANMS,
    'DS': DescendingScore,
}


class FeatureSelector():
    """ Selects features from an image using different filters.
    eig0: The smallest eigenvalue of the structure tensor.
    harris: The Harris corner response function.
    trigs: The Triggs corner response function.
    harmonic_mean: The harmonic mean of the eigenvalues of the structure tensor.

    Args:
        image (ndarray): The image to select features from.
    """
    def __init__(self, image, verbose=True) -> None:
        self.image = image
        self.roi_size   = (9, 9)
        self.score_image = None

        if verbose:
            print("\nAvailable Filters and the parameters they take:")
            for name, cls in available_filter_shortcuts.items():
                parameters = getattr(cls, 'parameters', [])
                filter_parameters = getattr(cls, 'filter_parameters', [])
                param_str = f"({', '.join(parameters)})" if parameters else ""
                filter_param_str = f"({', '.join(filter_parameters)})" if filter_parameters else ""
                print(f"  '{name}': {cls.__name__} {param_str} {filter_param_str}")
            print("\nAvailable Pickers and the parameters they take:")
            for name, cls in available_pickers_shortcuts.items():
                parameters = getattr(cls, 'parameters', [])
                param_str = f"({', '.join(parameters)})" if parameters else ""
                print(f"  '{name}': {cls.__name__} {param_str}")
        
    def set_filter(self, name, **kwargs):
        name = name.upper()
        if name not in available_filter_shortcuts:
            raise ValueError(f"Filter '{name}' not recognized. Available: {list(available_filter_shortcuts.keys())}")
        
        filter_class = available_filter_shortcuts[name]
        
        # Extract and remove 'roi_size' if present
        roi_size = kwargs.pop('roi_size', None)
        if roi_size is not None:
            self.set_roi(roi_size)

        # Validate kwargs
        supported1 = getattr(filter_class, 'parameters', [])
        supported2 = getattr(filter_class, 'filter_parameters', [])
        supported = supported1 + supported2
        for k in kwargs:
            if k not in supported:
                raise TypeError(f"Parameter '{k}' not supported by filter '{name}'. Supported: {supported}")

        self.filter = filter_class(self.image, **kwargs)
    
    def set_picker(self, name, **kwargs):
        name = name.upper()
        if name not in available_pickers_shortcuts:
            raise ValueError(f"Picker '{name}' not recognized. Available: {list(available_pickers_shortcuts.keys())}")

        picker_class = available_pickers_shortcuts[name]
        supported = getattr(picker_class, 'parameters', [])
        for k in kwargs:
            if k not in supported:
                raise TypeError(f"Parameter '{k}' not supported by picker '{name}'. Supported: {supported}")

        self.picker = picker_class(**kwargs)    

    def set_roi(self, roi_size):
        if isinstance(roi_size, (int, float)):
            self.roi_size = (roi_size, roi_size)
        else:
            self.roi_size = roi_size

    #### Apply filter
    def apply_filter(self):
        if self.filter.n_layers == 0:
            # 2D filter window
            size = (self.roi_size[0], self.roi_size[1])
            row_of_interest = 0
        else:
            # 3D filter window
            size = (self.roi_size[0], self.roi_size[1], self.filter.n_layers)
        if self.filter.filter_parameters is None:
            score_image =  generic_filter(self.filter.to_filter, self.filter.filter, size=size)
        else:
            score_image =  generic_filter(self.filter.to_filter, self.filter.filter, size=size, extra_arguments = (self.filter.filter_parameters, ))

        # For 3D filters, slice the middle layer:
        if self.filter.n_layers > 0:
            row_of_interest = self.filter.n_layers // 2
            score_image = score_image[..., row_of_interest]
        
        score_image[np.isnan(score_image)] = 0
        self.score_image = score_image
        return
    
    #### Pick points
    def pick_points(self, score_image = None):
        if self.picker is None:
            raise RuntimeError("Picker not set. Use set_picker() first.")
        if score_image is not None:
            self.score_image = score_image
        elif self.score_image is None:
            self.apply_filter()
        points = self.picker.pick(self.score_image)
        return points