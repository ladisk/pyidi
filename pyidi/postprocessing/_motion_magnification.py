import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import scipy as sp
import imageio.v2 as iio

import os
import copy
from io import BytesIO
from typing import Union

from ..methods.idi_method import IDIMethod

def mode_shape_magnification(displacements: np.ndarray, 
                             magnification_factor: Union[int, float], 
                             idi: Union[IDIMethod, None] = None, 
                             image: Union[np.ndarray, np.memmap, None] = None,
                             points: Union[np.ndarray, None] = None,
                             background_brightness: float = 0.3,
                             show_undeformed: bool = False
                             ) -> np.ndarray:
    """
    Create an image of a magnified mode-shape of a structure. If a 'pyidi.pyIDI' 
    class instance is input as argument 'video', the argument 'image' is set to 
    'video.mraw[0]' and the argument 'points' is set to 'video.points'. These 
    values can be overwritten by specifying the 'image' and 'points' arguments 
    explicitly.

    :param displacements: displacement (mode-shape) vector
    :type displacements: numpy.ndarray
    :param magnification_factor: magnification factor
    :type magnification_factor: int or float
    :param video: pyIDI class instance,
        defaults to None
    :type video: pyidi.pyIDI or None, optional
    :param image: the reference image, on which mode-shape magnification is 
        performed, defaults to None
    :type image: numpy.ndarray, numpy.memmap or None, optional
    :param points: image coordinates, where displacements 'displacements' are 
        defined, defaults to None
    :type points: numpy.ndarray or None, optional
    :param background_brightness: brightness of the background, expected values
        in range [0, 1], defaults to 0.3
    :type background_brighness: float, optional
    :param show_undeformed: Show the reference image (argument 'image') 
        underneath the magnified mode-shape, defaults to False
    :type show_undeformed: bool, optional

    :return: image of a magnified mode-shape of the structure
    :rtype: numpy.ndarray
    """
    if hasattr(displacements, 'shape') and len(displacements.shape) == 2:
        pass
    else:
        raise TypeError("The expected data type for argument 'displacements' is"\
                        "a 2D array of image coordinates of points of interest.")
    
    if isinstance(magnification_factor, (int, float)):
        pass
    else:
        raise TypeError("Expected data type for argument 'magnification_factor'"\
                        " is int or float.")
    
    if (isinstance(background_brightness, (int, float)) and 
        0 <= background_brightness <= 1):
        pass
    else:
        raise TypeError("Expected data type for argument 'background_brightness'"\
                        " is float in range [0, 1].")
    
    if isinstance(show_undeformed, bool):
        pass
    else:
        raise TypeError("Expected data type for argument 'show_undeformed' is"\
                        " boolean.")
    
    if idi is not None:
        if image is not None:
            if isinstance(image, (np.ndarray, np.memmap)):
                img_in = image
            else:
                raise TypeError("Expected object types for argument 'image' are"\
                                " 'numpy.ndarray' and 'numpy.memmap'.")
        else:
            img_in = idi.video.mraw[0]

        if points is not None:
            if isinstance(points, np.ndarray):
                points = points
            else:
                raise TypeError("Expected object type for argument 'points' is "\
                                "'numpy.ndarray'.")
        else:
            points = idi.points

    elif image is not None and points is not None:
        if isinstance(image, (np.ndarray, np.memmap)):
            img_in = image
        else:
            raise TypeError("Expected object types for argument 'image' are "\
                            "'numpy.ndarray' and 'numpy.memmap'.")

        if isinstance(points, np.ndarray):
            points = points
        else:
            raise TypeError("Expected object type for argument 'points' is "\
                            "'numpy.ndarray'.")
    
    else:
        raise TypeError("Both the input image and the points of interest need "\
                        "to be input, either via 'video' attributes 'mraw' and"\
                        " 'points' or as seperate arguments 'image' and 'points'.")

    mesh, mesh_def = create_mesh(points = points,
                                 disp = displacements,
                                 mag_fact = magnification_factor)

    img_out, a, b = init_output_image(input_image = img_in, 
                                      mesh = mesh.points, 
                                      mesh_def = mesh_def.points,
                                      bb = background_brightness,
                                      bu = show_undeformed)
    
    res = warp_image_elements(img_in = img_in, 
                              img_out = img_out, 
                              mesh = mesh, 
                              mesh_def = mesh_def, 
                              a = a, 
                              b = b)

    return res

def animate(displacements: np.ndarray, 
            magnification_factor: Union[int, float], 
            idi: Union[IDIMethod, None] = None, 
            image: Union[np.ndarray, np.memmap, None] = None, 
            points: Union[np.ndarray, None] = None,
            fps: int = 30,
            n_periods: int = 3,
            filename: str = 'mode_shape_mag_video',
            output_format: str = 'gif', 
            background_brightness: float = 0.3,
            show_undeformed: bool = False
            )-> None:
    """
    Create a video of a magnified mode-shape of a structure. If a 'pyidi.pyIDI'
    class instance is input as argument 'video', the argument 'image' is set to 
    'video.mraw[0]' and the argument 'points' is set to 'video.points'. These 
    values can be overwritten by specifying the 'image' and 'points' arguments 
    explicitly.

    :param displacements: displacement vector
    :type displacements: numpy.ndarray
    :param magnification_factor: magnification factor
    :type magnification_factor: int or float
    :param video: pyIDI class instance, 
        defaults to None
    :type video: pyidi.pyIDI or None, optional
    :param image: the reference image, on which mode-shape magnification is 
        performed, defaults to None
    :type image: numpy.ndarray, numpy.memmap or None, optional
    :param points: image coordinates, where displacements 'displacements' are 
        defined, defaults to None
    :type points: numpy.ndarray or None, optional
    :param fps: framerate of the created video, 
        defaults to 30
    :type fps: int, optional
    :param n_periods: number of periods of oscilation to be animated,
        defaults to 3
    :type n_periods: int, optional
    :param filename: the name of the output video file
        defaults to 'mode_shape_mag_video'
    :type filename: str
    :param output_format: output format of the video, selected from 'gif', 'mp4', 
        'avi', 'mov', defaults to 'gif'
    :type output_format: str, optional
    :param background_brightness: brightness of the background, expected values
        in range [0, 1], defaults to 0.3
    :type background_brighness: float, optional
    :param show_undeformed: Show the reference image (argument 'image') 
        underneath the magnified mode-shape, defaults to True
    :type show_undeformed: bool, optional
    """
    if hasattr(displacements, 'shape') and len(displacements.shape) == 2:
        pass
    else:
        raise TypeError("The expected data type for argument 'displacements' is"\
                        " a 2D array of image coordinates of points of interest.")
    
    if isinstance(magnification_factor, (int, float)):
        pass
    else:
        raise TypeError("Expected data type for argument 'magnification_factor'"\
                        " is int or float.")
    
    if isinstance(fps, int):
        pass
    else:
        raise TypeError("Expected data type for argument 'fps' is int.")
    
    if isinstance(n_periods, int):
        pass
    else:
        raise TypeError("Expected data type for argument 'n_periods' is int.")
    
    if isinstance(filename, str):
        pass
    else:
        raise TypeError("Expected data type for argument 'filename' is str.")
    
    if isinstance(output_format, str):
        if output_format in ['gif', 'mp4', 'avi', 'mov']:
            pass
        else:
            raise ValueError("Expected value for argument 'output_format' is one"\
                             " of 'gif', 'mp4', 'avi', 'mov'.")
    else:
        raise TypeError("Expected data type for argument 'output_format' is str.")
    
    if (isinstance(background_brightness, float) and 
        0 <= background_brightness <= 1):
        pass
    else:
        raise TypeError("Expected data type for argument 'background_brightness'"\
                        " is float in range [0, 1].")
    
    if idi is not None:
        if image is not None:
            if isinstance(image, (np.ndarray, np.memmap)):
                img_in = image
            else:
                raise TypeError("Expected object types for argument 'image' are"\
                                " 'numpy.ndarray' and 'numpy.memmap'.")
        else:
            img_in = idi.video.mraw[0]

        if points is not None:
            if isinstance(points, np.ndarray):
                points = points
            else:
                raise TypeError("Expected object type for argument 'points' is "\
                                "'numpy.ndarray'.")
        else:
            points = idi.points

    elif image is not None and points is not None:
        if isinstance(image, (np.ndarray, np.memmap)):
            img_in = image
        else:
            raise TypeError("Expected object types for argument 'image' are "\
                            "'numpy.ndarray' and 'numpy.memmap'.")

        if isinstance(points, np.ndarray):
            points = points
        else:
            raise TypeError("Expected object type for argument 'points' is "\
                            "'numpy.ndarray'.")
    
    else:
        raise TypeError("Both the input image and the points of interest need "\
                        "to be input, either via 'video' attributes 'mraw' and"\
                        " 'points' or as seperate arguments 'image' and 'points'.")
    
    # Create subfolder defined in 'filename' argument (if needed)
    folder, name = os.path.split(filename)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    
    mesh, mesh_def = create_mesh(points = points,
                                 disp = displacements,
                                 mag_fact = magnification_factor)
    
    mesh_def_negative = create_mesh(points = points,
                                    disp = displacements,
                                    mag_fact = -magnification_factor)[1]
    
    # All frames of the output video are the same size, defined by the maximum
    # deflections
    img_out, a, b = init_output_image(input_image = img_in,
                                      mesh = mesh.points,
                                      mesh_def = np.concatenate((
                                          mesh_def.points,
                                          mesh_def_negative.points
                                      )),
                                      bb = background_brightness,
                                      bu = show_undeformed)
    
    # Harmonic oscilation 
    frames = np.linspace(0, 2 * np.pi * n_periods, fps * n_periods)
    amp = np.sin(frames) * magnification_factor

    if output_format == 'gif':
        temp_writer = iio.get_writer(uri = f'{filename}.{output_format}',
                                     mode = 'I',
                                     duration = 1)
        
    else:
        temp_writer = iio.get_writer(uri = f'{filename}.{output_format}',
                                     fps = fps)

    with temp_writer as writer:
        for i, el in enumerate(amp):
            try:
                img_out_i = copy.deepcopy(img_out)

                # Create the deformed mesh for a given frame
                mesh_def = create_mesh(points = points,
                                       disp = displacements,
                                       mag_fact = el)[1]

                res = warp_image_elements(img_in = img_in,
                                          img_out = img_out_i,
                                          mesh = mesh,
                                          mesh_def = mesh_def,
                                          a = a,
                                          b = b)
                
                fig, ax = plt.subplots()
                ax.imshow(res, 'gray')
                ax.axis('off')

                buffer = BytesIO()
                fig.savefig(buffer,
                            format = 'png',
                            bbox_inches = 'tight',
                            transparent = True,
                            pad_inches = 0
                            )
                buffer.seek(0)

                figure = iio.imread(buffer)
                writer.append_data(figure)

                plt.close()

            except ValueError:
                writer.close()
                print("Failed to generate entire video, try a smaller magnification"\
                        " factor.")
                break

    buffer.close()
    writer.close()

    print(f'Video saved in file: {filename}.{output_format}')
        

def create_mesh(points, disp, mag_fact):
    """
    Generates a planar mesh of triangles based on the input set of points. Then 
    generates the deformed planar mesh of triangles based on the displacement 
    vectors 'disp', scaled by the magnification factor 'mag_fact'.
    """
    
    # Create undeformed mesh
    mesh = sp.spatial.Delaunay(points)

    # Create deformed mesh
    # The coordinates of the original mesh are over-written with their counter-
    # parts in the warped mesh, while the triangle connectivity of the original
    # mesh is retained.
    mesh_def = copy.deepcopy(mesh)
    mesh_def.points[:, 0] -= disp[:, 0] * mag_fact
    mesh_def.points[:, 1] += disp[:, 1] * mag_fact
    
    return mesh, mesh_def

def init_output_image(input_image, mesh, mesh_def, bb, bu):
    """
    Initialze the output image. The output image needs to be large enough to 
    prevent clipping of the motion magnified shape.
    """

    d = np.array([
          np.min(mesh[:, 1]) - np.min(mesh_def[:, 1]),
        - np.max(mesh[:, 1]) + np.max(mesh_def[:, 1]),
          np.min(mesh[:, 0]) - np.min(mesh_def[:, 0]),
        - np.max(mesh[:, 0]) + np.max(mesh_def[:, 0])
    ])
    d = np.round(d).astype('int')

    a = np.max(np.abs([d[2], d[3]]))
    b = np.max(np.abs([d[0], d[1]]))

    val = np.average(input_image) * bb
    if bu:
        out = cv.copyMakeBorder(input_image * bb,
                                top = a,
                                bottom = a,
                                left = b,
                                right = b,
                                borderType = cv.BORDER_CONSTANT,
                                value = val)
    else:
        out = np.ones((input_image.shape[0] + 2 * a, 
                       input_image.shape[1] + 2 * b)) * val
    return out, a, b

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
