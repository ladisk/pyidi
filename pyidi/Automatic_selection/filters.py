import numpy as np

#### Base Class
class GradientFilterBase:
    """
    Base class for filters operating on image gradients.

    Attributes:
        image (ndarray): Input image.
        Gi (ndarray): Gradient in the i (row) direction.
        Gj (ndarray): Gradient in the j (column) direction.
        to_filter (ndarray): Stack of gradient vectors (shape: HxWx2) as [-Gi, Gj].
        n_layers (int): Number of layers (used for filtering, typically 2).
        parameter: Any static parameter for preprocessing.
        filter_parameter: Parameters required during the filtering stage.

    Class Attributes:
        parameters (list): Parameters passed at construction time.
        filter_parameters (list): Parameters passed to the filter() method at runtime.
    """
    def __init__(self, image):
        self.image = image
        self.Gi, self.Gj = np.gradient(image)
        self.to_filter = np.transpose(np.array([-self.Gi, self.Gj]), (1, 2, 0))
        self.n_layers = 2
        self.parameters = None
        self.filter_parameters = None


#### Shi–Tomasi
class ShiTomasi(GradientFilterBase):
    """
    Implements the Shi–Tomasi corner score.

    Class Attributes:
        parameters (list): No constructor parameters required.
        filter_parameters (list): No filter parameters required.

    Methods:
        filter(pixel_list): Computes score using minimum eigenvalue calculation.
    """
    parameters = []
    filter_parameters = []

    def filter(self, pixel_list):
        ATA00 = pixel_list[::2] @ pixel_list[::2]
        ATA01 = pixel_list[::2] @ pixel_list[1::2]
        ATA11 = pixel_list[1::2] @ pixel_list[1::2]
        m = (ATA00 + ATA11) / 2
        p = ATA00 * ATA11 - ATA01**2
        return m - np.sqrt(m**2 - p)


#### DirectionalFilter
class DirectionalFilter:
    """
    Computes directional image response along a specific vector.

    Args:
        image (ndarray): Input image.
        dij (tuple): Direction vector (normalized internally).
        c (float): Weight for subtracting orthogonal component.

    Attributes:
        to_filter (ndarray): Preprocessed directional response.
        n_layers (int): Number of layers (0 as it works on scalar image).
        parameter (None): Placeholder for compatibility.

    Class Attributes:
        parameters (list): ['dij', 'c']
        filter_parameters (list): []

    Methods:
        filter(pixel_list): Returns average value of directional response window.
    """
    parameters = ['dij', 'c']
    filter_parameters = []

    def __init__(self, image, dij=(0, 1), c=0):
        self.image = image
        self.Gi, self.Gj = np.gradient(image)
        dij = np.array(dij, dtype=float)
        dij /= np.linalg.norm(dij)
        self.to_filter = np.abs(self.Gi * dij[0] + self.Gj * dij[1]) - c * np.abs(self.Gi * dij[1] + self.Gj * dij[0])
        self.n_layers = 0
        self.parameters = [dij, c]
        self.filter_parameters = None

    def filter(self, pixel_list):
        return np.mean(pixel_list)


#### Harris
class Harris(GradientFilterBase):
    """
    Implements Harris corner detection using the determinant and trace of the gradient structure tensor.

    Args:
        image (ndarray): Input image.
        alpha (float): Harris constant controlling sensitivity to corners.

    Class Attributes:
        parameters (list): []
        filter_parameters (list): ['alpha']

    Methods:
        filter(pixel_list, alpha): Returns Harris corner strength.
    """
    parameters = []
    filter_parameters = ['alpha']

    def __init__(self, image, alpha=0.04):
        super().__init__(image)
        self.filter_parameters = alpha

    def filter(self, pixel_list, alpha):
        ATA00 = pixel_list[::2] @ pixel_list[::2]
        ATA01 = pixel_list[::2] @ pixel_list[1::2]
        ATA11 = pixel_list[1::2] @ pixel_list[1::2]
        m = (ATA00 + ATA11) / 2
        p = ATA00 * ATA11 - ATA01**2
        return p - alpha * (m**2)


#### Triggs
class Triggs(GradientFilterBase):
    """
    Implements the Triggs corner metric based on eigenvalue difference.

    Args:
        image (ndarray): Input image.
        alpha (float): Weight controlling relative importance of eigenvalues.

    Class Attributes:
        parameters (list): []
        filter_parameters (list): ['alpha']

    Methods:
        filter(pixel_list, alpha): Returns Triggs response.
    """
    parameters = []
    filter_parameters = ['alpha']

    def __init__(self, image, alpha=0.5):
        super().__init__(image)
        self.filter_parameters = alpha

    def filter(self, pixel_list, alpha):
        ATA00 = pixel_list[::2] @ pixel_list[::2]
        ATA01 = pixel_list[::2] @ pixel_list[1::2]
        ATA11 = pixel_list[1::2] @ pixel_list[1::2]
        m = (ATA00 + ATA11) / 2
        p = ATA00 * ATA11 - ATA01**2
        sqrt_term = np.sqrt(m**2 - p)
        lam0 = m - sqrt_term
        lam1 = m + sqrt_term
        return lam0 - lam1 * alpha


#### Harmonic Mean
class HarmonicMean(GradientFilterBase):
    """
    Computes a harmonic mean-like response using the gradient structure tensor.

    Class Attributes:
        parameters (list): []
        filter_parameters (list): []

    Methods:
        filter(pixel_list): Returns harmonic corner measure.
    """
    parameters = []
    filter_parameters = []

    def filter(self, pixel_list):
        ATA00 = pixel_list[::2] @ pixel_list[::2]
        ATA01 = pixel_list[::2] @ pixel_list[1::2]
        ATA11 = pixel_list[1::2] @ pixel_list[1::2]
        numerator = ATA00 * ATA11 - ATA01**2
        denominator = ATA00 + ATA11
        return numerator / denominator if denominator != 0 else 0.0
