import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import imageio
import scipy as sp
import copy

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
    img_in = video.mraw[0]

    mesh = generate_planar_mesh(video.points)

    mesh_def = warp_mesh(mesh=mesh, disp=disp, mag_fact=mag_fact)

    img_out, a, b = init_output_image(img_in, video.points, mesh_def)
    
    res = warp_image_elements(img_in, img_out, mesh, mesh_def, a, b)

    return res

def animate(input_image, n_frames = 30, dpi = 100):
    """
    Create EMA based motion magnified video.

    :param input_image: Input image to be warped to produce the motion magnified
    video
    :type input_image: numpy.ndarray
    :param n_frames: Number of frames to be generated, default = 30
    :type n_frames: Int
    :param dpi: Dots per inch, used to control the quality of the out-put video,
    default = 100
    :type dpi: Int
    :return:
    :rtype:
    """

    pass

def generate_planar_mesh(points):
    """
    Generate a planar mesh of triangles from input points (scipy version).

    :param points: Input points for mesh generation, 
                given by pairs of coordinates (x, y).
    :type points: numpy.ndarray
    :return: Planar triangle mesh
    :rtype: scipy.spatial.qhull.Delaunay Class Instance
    """

    # Switch x and y columns
    pts = np.column_stack((points[:,0], 
                           points[:,1]))
    mesh = sp.spatial.Delaunay(pts)

    return mesh

def warp_mesh(mesh, disp, mag_fact):
    """
    Translate and warp mesh nodes based on displacements and magnification 
    factor.

    :param mesh: Input mesh
    :type mesh: scipy.spatial.qhull.Delaunay Class Instance
    :param disp: Displacements to be applied
    :type disp: numpy.ndarray
    :param mag_fact: Magnification factor
    :type mag_fact: positive int or float
    :return: Warped mesh
    :rtype: scipy.spatial.qhull.Delaunay Class Instance
    """

    # The coordinates of the original mesh are over-written with their counter-
    # parts in the warped mesh, while the triangle connectivity of the original
    # mesh is retained.

    mesh_def = copy.deepcopy(mesh)
    mesh_def.points[:,0] = mesh.points[:,0] - disp[:,0] * mag_fact
    mesh_def.points[:,1] = mesh.points[:,1] + disp[:,1] * mag_fact

    return mesh_def

def init_output_image(input_image, coord, warp):
    """
    Initialze the output image.

    :param input_image: Input image to be warped to produce the motion magnified
    image
    :type input_image: numpy.ndarray
    :param coord: Coordinates of points, where the displacement vectors are
    defined
    :type coord: numpy.ndarray
    :param warp: Warped planar mesh of triangles
    :type warp:pyvista.PolyData
    :return type: Output image with correct size to prevent clipping
    :rtype: numpy.ndarray
    """
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
    min_x = np.min(warp.points[:, 1])
    max_x = np.max(warp.points[:, 1])
    min_y = np.min(warp.points[:, 0])
    max_y = np.max(warp.points[:, 0])
    
    # Calculate the new size for the output image based on the minimum distance 
    # and mesh node coordinates
    new_width = int(max_x - min_x + 2 * min_distance)
    new_height = int(max_y - min_y + 2 * min_distance)

    dy = int(abs(min_distance - min_y))
    dx = int(abs(min_distance - min_x))

    out = np.ones((new_height, new_width)) * np.average(input_image) * 0.3
    # out[dy : dy + input_height, dx : dx + input_width] = input_image * 0.3
    
    return out, dy, dx

def warp_image_elements(img_in, img_out, mesh, mesh_def, a, b):
    """
    Warp image elements based on mesh and deformed mesh nodes.

    :param img_in: Input image
    :type img_in: numpy.ndarray
    :param img_out: Output image
    :type img_out: numpy.ndarray
    :param mesh: Original mesh
    :type mesh: pyvista.PolyData
    :param mesh_def: Deformed mesh
    :type mesh_def: pyvista.PolyData
    :param a: Offset value for y-axis
    :type a: int
    :param b: Offset value for x-axis
    :type b: int
    :return: Warped output image
    :rtype: numpy.ndarray
    """

    for i in range(len(mesh.simplices)):
        el_0 = np.float32(mesh.points[mesh.simplices[i]])
        el_1 = np.float32(mesh_def.points[mesh.simplices[i]])

        rect_0 = cv.boundingRect(el_0)
        rect_1 = cv.boundingRect(el_1)

        reg_0 = [((el_0[j, 1] - rect_0[1]), 
                  (el_0[j, 0] - rect_0[0])) 
                    for j in range(3)]

        reg_1 = [((el_1[j, 1] - rect_1[1]),
                  (el_1[j, 0] - rect_1[0])) 
                    for j in range(3)]

        crop_0 = img_in[rect_0[0] : rect_0[0] + rect_0[2],
                        rect_0[1] : rect_0[1] + rect_0[3]]

        aff_mat = cv.getAffineTransform(
            src = np.float32(reg_0),
            dst = np.float32(reg_1)
        )

        crop_1 = cv.warpAffine(
            src = crop_0,
            M = aff_mat,
            dsize = (rect_1[3], rect_1[2]),
            dst = None,
            flags = cv.INTER_LINEAR,
            borderMode = cv.BORDER_REFLECT_101,
        )

        mask = np.zeros((rect_1[2], rect_1[3]), dtype=np.float32)
        mask = cv.fillConvexPoly(
            img = mask,
            points = np.int32(reg_1),
            color = 1,
            lineType = cv.LINE_AA,
            shift=0
        )

        img_out[
            rect_1[0] + a : rect_1[0] + rect_1[2] + a,
            rect_1[1] + b : rect_1[1] + rect_1[3] + b
        ] = img_out[
            rect_1[0] + a : rect_1[0] + rect_1[2] + a,
            rect_1[1] + b : rect_1[1] + rect_1[3] + b
        ] * (1.0 - mask) + crop_1 * mask

    return img_out


# def generate_planar_mesh(points):
#     """
#     Generate a planar mesh of triangles from input points.

#     :param points: Input points for mesh generation, 
#                 given by pairs of coordinates.
#     :type points: numpy.ndarray
#     :return: Planar triangle mesh
#     :rtype: pyvista.PolyData
#     """
#     # Create a planar mesh of triangles from the input points
#     mesh = pv.PolyData(
#         np.column_stack((points[:,1], 
#                          points[:,0], 
#                          np.zeros(points.shape[0]))))
    
#     mesh = mesh.delaunay_2d()

#     return mesh

# def warp_mesh(mesh, disp, mag_fact):
#     """
#     Translate and warp mesh nodes based on displacements and magnification 
#     factor.

#     :param mesh: Input mesh
#     :type mesh: pyvista.PolyData
#     :param disp: Displacements to be applied
#     :type disp: numpy.ndarray
#     :param mag_fact: Magnification factor
#     :type mag_fact: positive int or float
#     :return: Warped mesh
#     :rtype: pyvista.PolyData
#     """

#     # Translate the mesh nodes in accordance with "disp", scaled by "mag_fact"
#     vect = np.column_stack((disp[:,1], 
#                             - disp[:,0], 
#                             np.zeros(disp.shape[0])))

#     mesh.add_field_data(vect, "vectors")
#     mesh_def = mesh.warp_by_vector(vectors = "vectors", factor = mag_fact)

#     return mesh_def

# def warp_image_elements(img_in, img_out, mesh, mesh_def, a, b):
#     """
#     Warp image elements based on mesh and deformed mesh nodes.

#     :param img_in: Input image
#     :type img_in: numpy.ndarray
#     :param img_out: Output image
#     :type img_out: numpy.ndarray
#     :param mesh: Original mesh
#     :type mesh: pyvista.PolyData
#     :param mesh_def: Deformed mesh
#     :type mesh_def: pyvista.PolyData
#     :param a: Offset value for y-axis
#     :type a: int
#     :param b: Offset value for x-axis
#     :type b: int
#     :return: Warped output image
#     :rtype: numpy.ndarray
#     """

#     for i in range(mesh.n_cells):
#         el_0 = np.float32(mesh.cell_points(i)[:,:2])
#         el_1 = np.float32(mesh_def.cell_points(i)[:,:2])

#         rect_0 = cv.boundingRect(el_0)
#         rect_1 = cv.boundingRect(el_1)

#         reg_0 = [((el_0[j, 0] - rect_0[0]), 
#                     (el_0[j, 1] - rect_0[1])) 
#                     for j in range(3)]

#         reg_1 = [((el_1[j, 0] - rect_1[0]), 
#                     (el_1[j, 1] - rect_1[1])) 
#                     for j in range(3)]

#         crop_0 = img_in[rect_0[1] : rect_0[1] + rect_0[3],
#                         rect_0[0] : rect_0[0] + rect_0[2]]

#         aff_mat = cv.getAffineTransform(
#             src = np.float32(reg_0),
#             dst = np.float32(reg_1)
#         )

#         crop_1 = cv.warpAffine(
#             src = crop_0,
#             M = aff_mat,
#             dsize = (rect_1[2], rect_1[3]),
#             dst = None,
#             flags = cv.INTER_LINEAR,
#             borderMode = cv.BORDER_REFLECT_101,
#         )

#         mask = np.zeros((rect_1[3], rect_1[2]), dtype=np.float32)
#         mask = cv.fillConvexPoly(
#             img = mask,
#             points = np.int32(reg_1),
#             color = 1,
#             lineType = cv.LINE_AA,
#             shift=0
#         )

#         img_out[
#             rect_1[1] + a : rect_1[1] + rect_1[3] + a,
#             rect_1[0] + b : rect_1[0] + rect_1[2] + b
#         ] = img_out[
#             rect_1[1] + a : rect_1[1] + rect_1[3] + a,
#             rect_1[0] + b : rect_1[0] + rect_1[2] + b
#         ] * (1.0 - mask) + crop_1 * mask

#     return img_out