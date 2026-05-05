.. _implemented_disp_id_methods:

Displacement identification methods
===============================================

Simplified Optical Flow (SOF)
-----------------------------

    [1] Javh, J., Slavič, J., & Boltežar, M. (2017). The subpixel resolution of optical-flow-based modal analysis. Mechanical Systems and Signal Processing, 88, 89–99. https://doi.org/10.1016/j.ymssp.2016.11.009

Lucas-Kanade (LK)
-----------------

    [2] Lucas, B. D., & Kanade, T. (1981). An Iterative Image Registration Technique with an Application to Stereo Vision. In Proceedings of the 7th International Joint Conference on Artificial Intelligence - Volume 2 (pp. 674–679). San Francisco, CA, USA: Morgan Kaufmann Publishers Inc. Retrieved from http://dl.acm.org/citation.cfm?id=1623264.1623280

Directional DIC
------------------------

    [3] Masmeijer T., Habtour E., Zaletelj K. & Slavič J. (2025). Directional DIC method with automatic feature selection. Mechanical Systems and Signal Processing, 224. https://doi.org/10.1016/j.ymssp.2024.112080

Digital Image Correlation (DIC)
-------------------------------

Full-field 2D Digital Image Correlation method using Inverse Compositional Gauss-Newton
(IC-GN) optimization with the Zero Normalized Sum of Squared Differences (ZNSSD) criterion.

This method is a port of the **pyDIC** library by the LADISK research group
(University of Ljubljana, Faculty of Mechanical Engineering) into the pyidi
multi-point ``IDIMethod`` framework. The original implementation is available at
https://github.com/ladisk/pyDIC and provides the algorithmic basis for this
class (gradient kernel, Jacobians, steepest-descent images, Hessian,
inverse-compositional warp update, ZNSSD error image). Please cite both the
underlying algorithm and the pyDIC repository when using this method.

Two warp models are supported:

* ``warp='affine'`` (default, 6 parameters): full first-order shape function with
  translation, normal strains, shear and rotation. Parameter vector
  ``[du/dx, du/dy, u, dv/dx, dv/dy, v]``.
* ``warp='rigid'`` (3 parameters): translation and in-plane rotation. Parameter vector
  ``[u, v, phi]``.

In addition to the standard ``displacements`` array of shape ``(n_points, n_frames, 2)``,
the method exposes the full converged warp parameters as ``self.warp_params`` of shape
``(n_points, n_frames, n_param)``. From the affine parameters one can directly recover
in-plane strains and rotation, e.g.::

    eps_xx   = idi.warp_params[..., 0]
    eps_yy   = idi.warp_params[..., 4]
    shear_xy = 0.5 * (idi.warp_params[..., 1] + idi.warp_params[..., 3])
    rotation = 0.5 * (idi.warp_params[..., 3] - idi.warp_params[..., 1])  # rad

The implementation is a port of the pyDIC algorithm (https://github.com/ladisk/pyDIC)
into the pyidi multi-point method framework.

    [4] Baker, S., & Matthews, I. (2004). Lucas-Kanade 20 Years On: A Unifying Framework. International Journal of Computer Vision, 56(3), 221-255. https://doi.org/10.1023/B:VISI.0000011205.11775.fd

    [5] Pan, B., Qian, K., Xie, H., & Asundi, A. (2009). Two-dimensional digital image correlation for in-plane displacement and strain measurement: a review. Measurement Science and Technology, 20(6), 062001. https://doi.org/10.1088/0957-0233/20/6/062001