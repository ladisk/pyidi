import numpy as np

#### Filters
class ShiTomasi():
    def __init__(self, image) -> None:
        self.image = image
        self.Gi, self.Gj = np.gradient(image)
        self.to_filter = np.transpose(np.array([-self.Gi, self.Gj]), (1, 2, 0))
        self.n_layers = 2
        self.parameter = None
        return
    
    def filter(self, pixel_list): #eig0
        ATA00 = pixel_list[::2] @ pixel_list[::2]  # equivalent to ATA[0, 0]
        ATA01 = pixel_list[::2] @ pixel_list[1::2]  # equivalent to ATA[0, 1]
        ATA11 = pixel_list[1::2] @ pixel_list[1::2]  # equivalent to ATA[1, 1]
        m = (ATA00 + ATA11) / 2
        p = ATA00 * ATA11 - ATA01**2
        return m - np.sqrt(m**2 - p)


def harris_filter(pixel_list, alpha_harris): #harris
    ATA00 = pixel_list[::2] @ pixel_list[::2]
    ATA01 = pixel_list[::2] @ pixel_list[1::2]
    ATA11 = pixel_list[1::2] @ pixel_list[1::2] 
    m = (ATA00 + ATA11) / 2
    p = ATA00 * ATA11 - ATA01**2
    return p -  alpha_harris* (m**2)

def trigs_filter(pixel_list, alpha_trigs): #trigs
    ATA00 = pixel_list[::2] @ pixel_list[::2]
    ATA01 = pixel_list[::2] @ pixel_list[1::2]
    ATA11 = pixel_list[1::2] @ pixel_list[1::2] 
    m = (ATA00 + ATA11) / 2
    p = ATA00 * ATA11 - ATA01**2
    temp =  np.sqrt(m**2 - p)
    lam0 = m - temp
    lam1 = m + temp
    return lam0 - lam1 * alpha_trigs

def harmonic_mean_filter(pixel_list):
    ATA00 = pixel_list[::2] @ pixel_list[::2]
    ATA01 = pixel_list[::2] @ pixel_list[1::2]
    ATA11 = pixel_list[1::2] @ pixel_list[1::2]
    return (ATA00 * ATA11 - ATA01**2)/ (ATA00 + ATA11)