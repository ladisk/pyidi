import numpy as np


def _compute_displacement_along_gradient(image_ref, image, gradient_magnitude):
    """
    Compute the displacement of each point along the gradient direction.

    :param image_ref: reference subset mean of each point
    :type image_ref: numpy.ndarray
    :param image: current subset mean of each point
    :type image: numpy.ndarray
    :param gradient_magnitude: gradient magnitude of the reference image at
        each point
    :type gradient_magnitude: numpy.ndarray
    :return: displacement of each point along the gradient direction
    :rtype: numpy.ndarray
    """
    difference = image_ref - image
    displacement_along_gradient = difference / (gradient_magnitude)

    return displacement_along_gradient


def _compute_displacement_along_image_axis(displacement_along_gradient, sign_0, sign_1, direction_correction_0, direction_correction_1):
    """
    Project the displacement along the gradient onto the image axes.

    :param displacement_along_gradient: displacement of each point along the
        gradient direction
    :type displacement_along_gradient: numpy.ndarray
    :param sign_0: sign of the gradient in direction 0 at each point
    :type sign_0: numpy.ndarray
    :param sign_1: sign of the gradient in direction 1 at each point
    :type sign_1: numpy.ndarray
    :param direction_correction_0: absolute direction cosine of the gradient
        in direction 0 at each point
    :type direction_correction_0: numpy.ndarray
    :param direction_correction_1: absolute direction cosine of the gradient
        in direction 1 at each point
    :type direction_correction_1: numpy.ndarray
    :return: displacement of each point, in (y, x), of shape ``(n_points, 2)``
    :rtype: numpy.ndarray
    """
    corrected_displacement_0 = displacement_along_gradient * sign_0 * direction_correction_0
    corrected_displacement_1 = displacement_along_gradient * sign_1 * direction_correction_1

    displacement_along_image_axis = np.stack((corrected_displacement_0, corrected_displacement_1), axis=-1)

    return displacement_along_image_axis
