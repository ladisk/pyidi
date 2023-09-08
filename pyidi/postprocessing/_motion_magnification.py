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
    img_out, a, b = init_output_image(video.mraw[0], video.points, shell_def)

    # Element-wise image warping
    for i in range(shell.n_cells):
        el_0 = np.float32(shell.cell_points(i)[:,:2])
        el_1 = np.float32(shell_def.cell_points(i)[:,:2])

        rect_0 = cv.boundingRect(el_0)
        rect_1 = cv.boundingRect(el_1)

        reg_0 = [((el_0[j, 0] - rect_0[0]), 
                    (el_0[j, 1] - rect_0[1])) 
                    for j in range(3)]

        reg_1 = [((el_1[j, 0] - rect_1[0]), 
                    (el_1[j, 1] - rect_1[1])) 
                    for j in range(3)]

        crop_0 = video.mraw[0][rect_0[1] : rect_0[1] + rect_0[3],
                        rect_0[0] : rect_0[0] + rect_0[2]]

        aff_mat = cv.getAffineTransform(
            src = np.float32(reg_0),
            dst = np.float32(reg_1)
        )

        crop_1 = cv.warpAffine(
            src = crop_0,
            M = aff_mat,
            dsize = (rect_1[2], rect_1[3]),
            dst = None,
            flags = cv.INTER_LINEAR,
            borderMode = cv.BORDER_REFLECT_101,
        )

        mask = np.zeros((rect_1[3], rect_1[2]), dtype=np.float32)
        mask = cv.fillConvexPoly(
            img = mask,
            points = np.int32(reg_1),
            color = 1,
            lineType = cv.LINE_AA,
            shift=0
        )

        img_out[
            rect_1[1] + a : rect_1[1] + rect_1[3] + a,
            rect_1[0] + b : rect_1[0] + rect_1[2] + b
        ] = img_out[
            rect_1[1] + a : rect_1[1] + rect_1[3] + a,
            rect_1[0] + b : rect_1[0] + rect_1[2] + b
        ] * (1.0 - mask) + crop_1 * mask

    return img_out


def init_output_image(input_image, coord, vect):
    # Find the dimensions of the input image
    input_height, input_width = input_image.shape[:2]

    # Calculate the distances between mesh nodes and image edges
    distances = np.array([
        coord[:, 1],           # Distances to the left edge
        coord[:, 0],           # Distances to the top edge
        input_width - coord[:, 1],  # Distances to the right edge
        input_height - coord[:, 0]  # Distances to the bottom edge
    ])
    
    # Calculate the minimum distance from the edges
    min_distance = np.min(distances)
    
    # Calculate the minimum and maximum of the deformed mesh nodes
    min_x = np.min(vect.points[:,0])
    max_x = np.max(vect.points[:,0])
    min_y = np.min(vect.points[:,1])
    max_y = np.max(vect.points[:,1])
    
    # Calculate the new size for the output image based on the minimum distance and mesh node coordinates
    new_width = int(max_x - min_x + 2 * min_distance)
    new_height = int(max_y - min_y + 2 * min_distance)

    dy = int(abs(min_distance - min_y))
    dx = int(abs(min_distance - min_x))

    out = np.ones((new_height, new_width)) * np.average(input_image) * 0.3
    # out[dy : dy + input_height, dx : dx + input_width] = input_image * 0.3
    
    return out, dy, dx
    
