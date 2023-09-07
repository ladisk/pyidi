import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import cv2 as cv
import pyvista as pv
import imageio

def motion_magnification(video, disp, mag_fact):
    """
    Perform Experimental Modal analysis based motion magnification.

    :param video: pyIDI class instance
    :type video: object
    :param disp: displacements to be magnified
    :type disp: numpy.ndarray
    :param mag_fact: the scalar magnification factor
    :type mag_fact: positive int or float
    """

    # Create a planar mesh of triangles from the input points
    mesh = pv.PolyData(np.column_stack((video.points[:,1], video.points[:,0], np.zeros(video.points.shape[0]))))
    
    shell = mesh.delaunay_2d()

    # Translate the mesh nodes in accordance with "disp", scaled by "mag_fact"
    vect = np.column_stack((disp[:,0], disp[:,1], np.zeros(video.points.shape[0])))

    shell.add_field_data(vect, "vectors")
    shell_def = shell.warp_by_vector(vectors = "vectors", factor = mag_fact)

    # Initialize the output image
    img_out = video.mraw[0] * 0.5
    
    # element-wise transformation
    
