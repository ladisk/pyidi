import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import scipy as sp
import copy
from typing import Union
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyidi

def motion_magnification(disp: np.ndarray, 
                         mag_fact: Union[int, float], 
                         video: Union["pyidi.pyIDI", None] = None, 
                         img: Union[np.ndarray, np.memmap, None] = None,
                         pts: Union[np.ndarray, None] = None
                         ) -> np.ndarray:
    """
    Perform Experimental Modal Analysis based motion magnification. If a 'pyidi.
    pyIDI class instance is input as argument 'video', the argument 'img' is set
    to 'video.mraw[0]' and the argument 'pts' is set to 'video.points'. These 
    values can be overwritten by specifying the 'img' and 'pts' arguments 
    explicitly.

    :param disp: displacement vector
    :type disp: numpy.ndarray
    :param mag_fact: magnification factor
    :type mag_fact: int or float
    :param video: pyIDI class instance,
        defaults to None
    :type video: pyidi.pyIDI, optional
    :param img: the reference image, on which motion magnification is performed,
        defaults to None
    :type img: numpy.ndarray or numpy.memmap, optional
    :param pts: image coordinates, where displacements 'disp' are defined,
        defaults to None
    :type pts: numpy.ndarray, optional

    :return: motion magnified image of the structure
    :rtype: numpy.ndarray
    """
    if hasattr(disp, 'shape') and len(disp.shape) == 2:
        pass
    else:
        raise TypeError("The expected data type for argument 'disp' is a 2D "\
                        "array of image coordinates of points of interest.")
    
    if isinstance(mag_fact, (int, float)):
        pass
    else:
        raise TypeError("Expected data type for argument 'mag_fact' is int or "\
                        "float.")
    
    if video is not None:
        if img is not None:
            if isinstance(img, (np.ndarray, np.memmap)):
                img_in = img
            else:
                raise TypeError("Expected object types for argument 'img' are "\
                                "'numpy.ndarray' and 'numpy.memmap'.")
        else:
            img_in = video.mraw[0]

        if pts is not None:
            if isinstance(pts, np.ndarray):
                points = pts
            else:
                raise TypeError("Expected object type for argument 'pts' is "\
                                "'numpy.ndarray'.")
        else:
            points = video.points

    elif img is not None and pts is not None:
        if isinstance(img, (np.ndarray, np.memmap)):
            img_in = img
        else:
            raise TypeError("Expected object types for argument 'img' are "\
                            "'numpy.ndarray' and 'numpy.memmap'.")

        if isinstance(pts, np.ndarray):
            points = pts
        else:
            raise TypeError("Expected object type for argument 'pts' is "\
                            "'numpy.ndarray'.")
    
    else:
        raise TypeError("Both the input image and the points of interest need "\
                        "to be input, either via 'video' attributes 'mraw' and"\
                        " 'points' or as seperate arguments 'img' and 'pts'.")

    mesh, mesh_def = create_mesh(points = points,
                                 disp = disp,
                                 mag_fact = mag_fact)

    img_out, a, b = init_output_image(input_image = img_in, 
                                      coord = points, 
                                      warp = mesh_def)
    
    res = warp_image_elements(img_in = img_in, 
                              img_out = img_out, 
                              mesh = mesh, 
                              mesh_def = mesh_def, 
                              a = a, 
                              b = b)

    return res

def animate(disp: np.ndarray, 
            mag_fact: Union[int, float], 
            video = None, 
            img: Union[np.ndarray, np.memmap] = None, 
            pts: np.ndarray = None,
            n_frames: int = 30,
            filename: str = 'Motion_mag_video'
            )-> None:
    """
    Create a video based on the Experimental modal analysis motion magnification. 
    If a 'pyidi.pyIDI class instance is input as argument 'video', the argument 
    'img' is set to 'video.mraw[0]' and the argument 'pts' is set to 'video.points'. 
    These values can be overwritten by specifying the 'img' and 'pts' arguments 
    explicitly.

    :param disp: displacement vector
    :type disp: numpy.ndarray
    :param mag_fact: magnification factor
    :type mag_fact: int or float
    :param video: pyIDI class instance, 
        defaults to None
    :type video: pyidi.pyIDI, optional
    :param img: the reference image, on which motion magnification is performed,
        defaults to None
    :type img: numpy.ndarray or numpy.memmap, optional
    :param pts: image coordinates, where displacements 'disp' are defined,
        defaults to None
    :type pts: numpy.ndarray, optional
    :param n_frames: number of frames per period, 
        defaults to 30
    :type n_frames: int, optional
    :param filename: the name of the output video file
        defaults to 'Motion_mag_video'
    :type filename: str
    """
    if hasattr(disp, 'shape') and len(disp.shape) == 2:
        pass
    else:
        raise TypeError("The expected data type for argument 'disp' is a 2D "\
                        "array of image coordinates of points of interest.")
    
    if isinstance(mag_fact, (int, float)):
        pass
    else:
        raise TypeError("Expected data type for argument 'mag_fact' is int or "\
                        "float.")
    
    if isinstance(n_frames, int):
        pass
    else:
        raise TypeError("Expected data type for argument 'n_frames' is int.")
    
    if isinstance(filename, str):
        pass
    else:
        raise TypeError("Expected data type for argument 'filename' is str.")
    
    if video is not None:
        if img is not None:
            if isinstance(img, (np.ndarray, np.memmap)):
                img_in = img
            else:
                raise TypeError("Expected object types for argument 'img' are "\
                                "'numpy.ndarray' and 'numpy.memmap'.")
        else:
            img_in = video.mraw[0]

        if pts is not None:
            if isinstance(pts, np.ndarray):
                points = pts
            else:
                raise TypeError("Expected object type for argument 'pts' is "\
                                "'numpy.ndarray'.")
        else:
            points = video.points

    elif img is not None and pts is not None:
        if isinstance(img, (np.ndarray, np.memmap)):
            img_in = img
        else:
            raise TypeError("Expected object types for argument 'img' are "\
                            "'numpy.ndarray' and 'numpy.memmap'.")

        if isinstance(pts, np.ndarray):
            points = pts
        else:
            raise TypeError("Expected object type for argument 'pts' is "\
                            "'numpy.ndarray'.")
    
    else:
        raise TypeError("Both the input image and the points of interest need "\
                        "to be input, either via 'video' attributes 'mraw' and"\
                        " 'points' or as seperate arguments 'img' and 'pts'.")
    
    mesh, mesh_def = create_mesh(points = points,
                                 disp = disp,
                                 mag_fact = mag_fact)
    
    # All frames of the output video are the same size, defined by the maximum
    # deflections
    img_out, a, b = init_output_image(input_image = img_in,
                                      coord = points,
                                      warp = mesh_def)
    
    frames = np.linspace(0, 2 * np.pi, n_frames)
    amp = np.sin(frames) * mag_fact

    result = cv.VideoWriter(filename = f'{filename}.avi',
                            fourcc = cv.VideoWriter_fourcc(*'XVID'),
                            fps = n_frames,
                            frameSize = (img_out.shape[1], img_out.shape[0]),
                            isColor = False)

    for i, el in enumerate(amp):

        img_out_i = copy.deepcopy(img_out)

        # Create the deformed mesh for a given frame
        mesh_def = create_mesh(points = points,
                               disp = disp,
                               mag_fact = el)[1]

        res = warp_image_elements(img_in = img_in,
                                  img_out = img_out_i,
                                  mesh = mesh,
                                  mesh_def = mesh_def,
                                  a = a,
                                  b = b)
        
        # The OpenCV VideoWriter approach to video generation only works with 8-
        # images
        norm = (res - np.min(res)) / (np.max(res) - np.min(res))
        result.write((norm * 255).astype('uint8'))
        cv.imshow('Frame', res)

    result.release()
    cv.destroyAllWindows()

    print(f'Video saved in file: {filename}.avi')
        
        

def create_mesh(points, disp, mag_fact):
    """
    Generates a planar mesh of triangles based on the input set of points. Then 
    generates the deformed planar mesh of triangles based on the displacement 
    vectors "disp", scaled by the magnification factor "mag_fact".
    """

    # Switch x and y columns
    pts = np.column_stack((points[:,0], 
                           points[:,1]))
    
    # Create undeformed mesh
    mesh = sp.spatial.Delaunay(pts)

    # Create deformed mesh
    # The coordinates of the original mesh are over-written with their counter-
    # parts in the warped mesh, while the triangle connectivity of the original
    # mesh is retained.
    mesh_def = copy.deepcopy(mesh)
    mesh_def.points[:, 0] = mesh.points[:, 0] - disp[:, 0] * mag_fact
    mesh_def.points[:, 1] = mesh.points[:, 1] + disp[:, 1] * mag_fact

    return mesh, mesh_def


def init_output_image(input_image, coord, warp):
    """
    Initialze the output image. The output image needs to be large enough to 
    prevent clipping of the motion magnified shape.
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
    
    return out, dy, dx

def warp_image_elements(img_in, img_out, mesh, mesh_def, a, b):
    """
    Warp image elements based on mesh and deformed mesh nodes.
    """

    for i in range(len(mesh.simplices)):
        el_0 = np.float32(mesh.points[mesh.simplices[i]])
        el_1 = np.float32(mesh_def.points[mesh.simplices[i]])

        # Define axis-aligned bounding rectangle for given triangle element in 
        # its original and deformed state
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

        # Definition of the affine transformation matrix for the given triangle 
        # element
        aff_mat = cv.getAffineTransform(
            src = np.float32(reg_0),
            dst = np.float32(reg_1)
        )

        # Execution of the affine transformation
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

        # Assembly of the transformed element into the output image
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