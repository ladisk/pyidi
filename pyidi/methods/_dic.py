"""Full-field 2D Digital Image Correlation method for pyidi.

This module is a port of the pyDIC library by the LADISK research group
(University of Ljubljana, Faculty of Mechanical Engineering) into pyidi's
multi-point ``IDIMethod`` framework. The original pyDIC implementation lives
at:

    https://github.com/ladisk/pyDIC

Algorithmically, this module mirrors the original: Inverse Compositional
Gauss-Newton (IC-GN) optimization with the Zero Normalized Sum of Squared
Differences (ZNSSD) criterion, with both a 6-parameter affine and a
3-parameter rigid (translation + in-plane rotation) warp model. Per-point
precomputables (gradient, steepest-descent images, Hessian) and the warp
update equations follow pyDIC's ``py_dic.dic`` and ``py_dic.dic_tools``
modules.

Differences with respect to the upstream pyDIC implementation:

- The single-ROI, function-call API of pyDIC is wrapped into a multi-point
  ``IDIMethod`` subclass (``DIC``) so that many subsets can be tracked in a
  single analysis with the standard pyidi configuration / checkpointing /
  multiprocessing infrastructure.
- The full converged warp parameters are exposed per point per frame as
  ``self.warp_params`` (see the ``DIC`` class docstring).
- Subset coordinates are mean-centered so that, at the identity warp, the
  spline of the target frame is sampled exactly at each point's center.
"""

import numpy as np
import time
import datetime
import warnings

import scipy.signal
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from psutil import cpu_count
from .. import tools
from ..video_reader import VideoReader

from .idi_method import IDIMethod
from ..progress_bar import progress_bar, rich_progress_bar_setup
try:
    from qtpy.QtWidgets import QApplication
except ImportError:
    QApplication = None


class DIC(IDIMethod):
    """Full-field 2D Digital Image Correlation method using Inverse
    Compositional Gauss-Newton optimization with the Zero Normalized Sum
    of Squared Differences (ZNSSD) criterion.

    Origin
    ------
    This class is a port of the pyDIC library
    (https://github.com/ladisk/pyDIC, LADISK research group, University of
    Ljubljana) into the pyidi multi-point ``IDIMethod`` framework. The core
    math (gradient kernel, Jacobians, steepest-descent images, Hessian,
    inverse-compositional warp update, ZNSSD error image) follows pyDIC's
    ``py_dic.dic`` and ``py_dic.dic_tools`` modules. Please cite the
    underlying algorithm and the pyDIC repository when using this method.

    Outputs
    -------
    After ``get_displacements()`` two arrays are populated on the instance:

    - ``self.displacements``: shape ``(n_points, n_frames, 2)`` with columns
      ``[dy, dx]``. This is the standard pyidi output (subset-center
      translation only) and is the value returned by ``get_displacements``.
    - ``self.warp_params``: shape ``(n_points, n_frames, n_param)`` holding
      the full converged warp parameter vector per point per frame.

    The contents of ``warp_params`` depend on the chosen warp model:

    **Affine** (``warp='affine'``, default, ``n_param=6``):
    columns are ``[du/dx, du/dy, u, dv/dx, dv/dy, v]`` where ``u, v`` are the
    translations in x and y, and ``du/dx``, ``du/dy``, ``dv/dx``, ``dv/dy``
    are the in-plane displacement gradients (first-order shape function).
    Useful derived quantities::

        eps_xx     = warp_params[..., 0]                      # du/dx
        eps_yy     = warp_params[..., 4]                      # dv/dy
        shear_xy   = 0.5 * (warp_params[..., 1] + warp_params[..., 3])
        rotation   = 0.5 * (warp_params[..., 3] - warp_params[..., 1])  # rad

    Green-Lagrange strains follow the standard formulae::

        E_xx = du/dx + 0.5 * (du/dx**2 + dv/dx**2)
        E_yy = dv/dy + 0.5 * (du/dy**2 + dv/dy**2)
        E_xy = 0.5 * (du/dy + dv/dx + du/dx*du/dy + dv/dx*dv/dy)

    **Rigid** (``warp='rigid'``, ``n_param=3``):
    columns are ``[u, v, phi]`` where ``phi`` is the in-plane rotation in
    radians.

    Persistence
    -----------
    When ``get_displacements(autosave=True)`` is used, ``warp_params`` is
    pickled to ``warp_params.pkl`` next to ``results.pkl`` in the analysis
    folder and is automatically reattached by ``pyidi.load_analysis``.
    """

    def configure(
        self, roi_size=(21, 21), pad=2, max_nfev=100, tol=1e-6,
        int_order=3, warp='affine', prefilter_gauss=True,
        verbose=1, show_pbar=True, processes=1, resume_analysis=False,
        reference_image=0, frame_range='full',
    ):
        """
        Configure the DIC method.

        :param roi_size: (h, w) height and width of the region of interest.
            ROI dimensions should be odd numbers. Defaults to (21, 21).
        :type roi_size: tuple, list, int, optional
        :param pad: padding around the ROI in px (interpolation safety).
        :type pad: int, optional
        :param max_nfev: maximum number of IC-GN iterations per point per frame.
        :type max_nfev: int, optional
        :param tol: convergence threshold on the L2 norm of the parameter
            increment ``dp``.
        :type tol: float, optional
        :param int_order: interpolation spline order for the target image.
        :type int_order: int, optional
        :param warp: warp model name. Either ``'affine'`` (6 parameters) or
            ``'rigid'`` (3 parameters). Defaults to ``'affine'``.
        :type warp: str, optional
        :param prefilter_gauss: if True, use the Gauss-prefiltered finite
            difference kernel ``[-0.446, 0, 0.446]`` for the reference
            gradient. Otherwise use ``[-0.5, 0, 0.5]``.
        :type prefilter_gauss: bool, optional
        :param verbose: show text while running.
        :type verbose: int, optional
        :param show_pbar: show progress bar.
        :type show_pbar: bool, optional
        :param processes: number of processes to run.
        :type processes: int, optional
        :param resume_analysis: if True, the last analysis results are loaded
            and computation continues from the last computed time point.
        :type resume_analysis: bool, optional
        :param reference_image: reference image. Index of a frame, tuple
            (slice) or numpy.ndarray.
        :type reference_image: int or tuple or ndarray
        :param frame_range: part of the video to process. If ``'full'``, the
            full video is processed.
        :type frame_range: tuple or "full"
        """
        # Mirror LucasKanade: only overwrite attribute if the argument is not None.
        if pad is not None:
            self.pad = pad
        if max_nfev is not None:
            self.max_nfev = max_nfev
        if tol is not None:
            self.tol = tol
        if verbose is not None:
            self.verbose = verbose
        if show_pbar is not None:
            self.show_pbar = show_pbar
        if int_order is not None:
            self.int_order = int_order
        if prefilter_gauss is not None:
            self.prefilter_gauss = prefilter_gauss
        if processes is not None:
            self.processes = processes
        if resume_analysis is not None:
            self.resume_analysis = resume_analysis
        if reference_image is not None:
            self.reference_image = reference_image
        if frame_range is not None:
            self.frame_range = frame_range
        if roi_size is not None:
            if isinstance(roi_size, int):
                roi_size = [roi_size, roi_size]
            self.roi_size = np.array(roi_size, dtype=int)
        if warp is not None:
            if warp not in ('affine', 'rigid'):
                raise ValueError(f"Unknown warp model '{warp}'. Must be 'affine' or 'rigid'.")
            self.warp = warp

        self._set_frame_range()

    def _set_frame_range(self):
        """Set the range of the video to be processed."""
        self.step_time = 1

        if self.frame_range == 'full':
            self.start_time = 1
            self.stop_time = self.video.N

        elif isinstance(self.frame_range, tuple):
            if len(self.frame_range) >= 2:
                if self.frame_range[0] < self.frame_range[1] and self.frame_range[0] >= 0:
                    self.start_time = self.frame_range[0] + self.step_time

                    if self.frame_range[1] <= self.video.N:
                        self.stop_time = self.frame_range[1]
                    else:
                        raise ValueError(
                            f'frame_range can only go to end of video - up to index '
                            f'{self.video.N}. selected range was: {self.frame_range}'
                        )
                else:
                    raise ValueError('Wrong frame_range definition.')

                if len(self.frame_range) == 3:
                    self.step_time = self.frame_range[2]

            else:
                raise Exception('Wrong definition of frame_range.')
        else:
            raise TypeError(
                f'frame_range must be a tuple of start and stop index or "full" '
                f'({type(self.frame_range)})'
            )

        self.N_time_points = len(range(self.start_time - self.step_time, self.stop_time, self.step_time))

    def calculate_displacements(self):
        """
        Calculate displacements for the configured points and ROI size.

        Convention notes
        ----------------
        - pyidi points are stored as ``(y, x)`` (row, column).
        - The warp matrix ``W`` is a 3x3 homogeneous transform whose first row
          drives the **x** coordinate and second row the **y** coordinate. The
          subset grid coordinates ``(xx, yy)`` are centered on zero, so when
          ``W = I`` the spline is sampled exactly at each point's center.
        - After convergence, ``W[0, 2]`` is the subpixel displacement in **x**
          and ``W[1, 2]`` is the subpixel displacement in **y**. These are
          stored as ``[dy, dx]`` in ``self.displacements``, matching the
          LucasKanade output convention.
        """
        video = self.video

        self._announce_run_state()

        if self.processes != 1:
            self._run_multiprocessing()
            return

        self.image_size = (video.image_height, video.image_width)
        self._init_storage()

        self.warnings = []
        start_time = time.time()

        if self.verbose:
            t = time.time()
            print('Precomputing reference quantities...')
        self._precompute_reference(video)
        if self.verbose:
            print(f'...done in {time.time() - t:.2f} s')

        # Per-point current warp matrices, warm-started across frames.
        # TODO: optionally pre-seed with FFT cross-correlation initial guess for the first frame.
        current_warps = [np.eye(3) for _ in range(self.points.shape[0])]

        self._run_time_loop(video, current_warps)

        del self.temp_disp

        if self.verbose:
            self._print_elapsed(time.time() - start_time)

        if self.process_number == 0:
            self.clear_temp_files()

    def _announce_run_state(self):
        """Print resume / new-analysis banner and reset ``resume_analysis`` if needed."""
        if self.process_number != 0:
            return
        if self.resume_analysis and self.temp_files_check():
            if self.verbose:
                print('-- Resuming last analysis ---')
                print(' ')
        else:
            self.resume_analysis = False
            if self.verbose:
                print('--- Starting new analysis ---')
                print(' ')

    def _run_multiprocessing(self):
        """Run the analysis using multiple processes."""
        if not self.resume_analysis:
            self.create_temp_files(init_multi=True)

        self.displacements, self.warp_params = multi(
            self.video, self, self.processes,
            configuration_keys=self.configuration_keys,
        )

        self.clear_temp_files()

    def _init_storage(self):
        """Allocate (or resume) the displacement and warp-parameter buffers."""
        n_param = 6 if self.warp == 'affine' else 3
        if self.resume_analysis:
            self.resume_temp_files()
            if not hasattr(self, 'warp_params'):
                self.warp_params = np.zeros(
                    (self.points.shape[0], self.N_time_points, n_param)
                )
        else:
            self.displacements = np.zeros((self.points.shape[0], self.N_time_points, 2))
            self.warp_params = np.zeros((self.points.shape[0], self.N_time_points, n_param))
            self.create_temp_files(init_multi=False)

    def _run_time_loop(self, video, current_warps):
        """Iterate over frames, optimizing each point and storing results."""
        len_of_task = len(range(self.start_time, self.stop_time, self.step_time))
        for ii, i in enumerate(progress_bar(self.start_time, self.stop_time, self.step_time)):
            if self.resume_analysis and hasattr(self, "completed_points") and self.completed_points > ii:
                continue
            ii = ii + 1

            G = video.get_frame(i).astype(np.float64)
            G_spline = RectBivariateSpline(
                np.arange(G.shape[0]), np.arange(G.shape[1]), G,
                kx=self.int_order, ky=self.int_order, s=0,
            )

            self._process_frame(G_spline, current_warps, i, ii)

            self.temp_disp[:, ii, :] = self.displacements[:, ii, :]
            self.update_log(ii)

            if hasattr(self, "progress") and hasattr(self, "task_id"):
                self.progress[self.task_id] = {"progress": ii + 1, "total": len_of_task}
            if QApplication is not None and QApplication.instance() is not None:
                QApplication.processEvents()

    def _process_frame(self, G_spline, current_warps, frame_index, ii):
        """Optimize all points for a single frame and store outputs."""
        for p, point in enumerate(self.points):
            W_p = self._optimize_warp(
                G_spline=G_spline,
                point=point,
                W_init=current_warps[p],
                point_index=p,
                frame=frame_index,
            )
            current_warps[p] = W_p

            self.displacements[p, ii, :] = [W_p[1, 2], W_p[0, 2]]
            if self.warp == 'affine':
                self.warp_params[p, ii, :] = _params_from_affine_matrix(W_p)
            else:
                self.warp_params[p, ii, :] = _params_from_rigid_matrix(W_p)

    @staticmethod
    def _print_elapsed(full_time):
        """Print the elapsed wall-clock time in min/s or s."""
        if full_time > 60:
            full_time_m = full_time // 60
            full_time_s = full_time % 60
            print(f'Time to complete: {full_time_m:.0f} min, {full_time_s:.1f} s')
        else:
            print(f'Time to complete: {full_time:.1f} s')

    def _optimize_warp(self, G_spline, point, W_init, point_index, frame):
        """
        Run IC-GN optimization for a single point at the current frame.

        :param G_spline: spline of the target frame ``G`` evaluated in image
            coordinates.
        :type G_spline: scipy.interpolate.RectBivariateSpline
        :param point: ``(y, x)`` image-space center of the subset.
        :type point: array_like of size 2
        :param W_init: initial 3x3 warp matrix (subset-local coordinates).
        :type W_init: numpy.ndarray
        :param point_index: index of the current point (for diagnostics).
        :type point_index: int
        :param frame: current frame index (for diagnostics).
        :type frame: int
        :return: converged 3x3 warp matrix.
        :rtype: numpy.ndarray
        """
        F = self._F_cache[point_index]
        F_mean = self._F_mean_cache[point_index]
        F_std = self._F_std_cache[point_index]
        SD = self._sd_cache[point_index]
        inv_H = self._inv_H_cache[point_index]
        yy = self._subset_yy
        xx = self._subset_xx

        py, px = float(point[0]), float(point[1])
        W = W_init.copy()

        for _ in range(self.max_nfev):
            # Apply warp to centered subset coords -> image-space sample coords.
            Xs = W[0, 0] * xx + W[0, 1] * yy + W[0, 2]
            Ys = W[1, 0] * xx + W[1, 1] * yy + W[1, 2]
            G_warped = G_spline.ev(py + Ys, px + Xs).reshape(F.shape)

            G_mean = G_warped.mean()
            G_std = G_warped.std()
            if G_std < 1e-10:
                self.warnings.append(
                    f'Point {point_index} frame {frame}: uniform G subset (std<1e-10), '
                    f'aborting iteration.'
                )
                break

            e = (F - F_mean) - (F_std / G_std) * (G_warped - G_mean)
            b = np.einsum('ihw,hw->i', SD, e)
            dp = inv_H @ b

            if self.warp == 'affine':
                W_dp = _affine_warp_matrix(dp)
            else:
                W_dp = _rigid_warp_matrix(dp)
            try:
                W_dp_inv = np.linalg.inv(W_dp)
            except np.linalg.LinAlgError:
                self.warnings.append(
                    f'Point {point_index} frame {frame}: singular incremental warp, '
                    f'aborting iteration.'
                )
                break

            W = W @ W_dp_inv

            if np.linalg.norm(dp) < self.tol:
                return W

        return W

    def _padded_slice(self, point, roi_size, image_shape, pad=None):
        """Return ``(yslice, xslice)`` cropping an image around ``point``.

        If the resulting slice would be out of bounds of the image (given by
        ``image_shape``), the slice is shifted to the image edge and a warning
        is issued.

        :param point: center coordinate of the desired ROI, ``(y, x)``.
        :type point: array_like of size 2
        :param roi_size: size of the cropped image, ``(h, w)``.
        :type roi_size: array_like of size 2
        :param image_shape: shape of the image to be sliced, ``(h, w)``.
        :type image_shape: array_like of size 2
        :param pad: pad border size in pixels. If None, ``self.pad`` is used.
        :type pad: int, optional
        :return: ``(yslice, xslice)`` to use for image slicing.
        :rtype: tuple of slice
        """
        if pad is None:
            pad = self.pad
        y_, x_ = np.array(point).astype(int)
        h, w = np.array(roi_size).astype(int)

        y = np.clip(y_, h // 2 + pad, image_shape[0] - (h // 2 + pad + 1))
        x = np.clip(x_, w // 2 + pad, image_shape[1] - (w // 2 + pad + 1))

        if x != x_ or y != y_:
            warnings.warn(
                'Reached image edge. The displacement optimization '
                'algorithm may not converge, or selected points might be too close '
                'to image border. Please check analysis settings.'
            )

        yslice = slice(y - h // 2 - pad, y + h // 2 + pad + 1)
        xslice = slice(x - w // 2 - pad, x + w // 2 + pad + 1)
        return yslice, xslice

    def _set_reference_image(self, video: VideoReader, reference_image):
        """Set the reference image."""
        if isinstance(reference_image, int):
            ref = video.get_frame(reference_image).astype(float)

        elif isinstance(reference_image, tuple):
            if len(reference_image) == 2:
                ref = np.zeros((video.image_height, video.image_width), dtype=float)
                for frame in range(reference_image[0], reference_image[1]):
                    ref += video.get_frame(frame)
                ref /= (reference_image[1] - reference_image[0])

        elif isinstance(reference_image, np.ndarray):
            ref = reference_image

        else:
            raise Exception('reference_image must be index of frame, tuple (slice) or ndarray.')

        return ref

    def _precompute_reference(self, video: VideoReader):
        """Precompute per-point reference quantities for the IC-GN loop.

        For every point this stores the reference subset ``F``, its mean and
        standard deviation, the steepest-descent images ``SD``, and the
        inverse Hessian ``inv_H``.

        :param video: parent video object.
        :type video: VideoReader
        """
        pad = self.pad
        f = self._set_reference_image(video, self.reference_image)
        h, w = int(self.roi_size[0]), int(self.roi_size[1])

        # Centered subset coordinate grid (h, w), reused for all points.
        yy, xx = np.meshgrid(
            np.arange(h) - h // 2,
            np.arange(w) - w // 2,
            indexing='ij',
        )
        self._subset_yy = yy.astype(np.float64)
        self._subset_xx = xx.astype(np.float64)

        if self.warp == 'affine':
            jac = _jacobian_affine(h, w)
        else:
            jac = _jacobian_rigid(h, w)

        F_cache = []
        F_mean_cache = []
        F_std_cache = []
        sd_cache = []
        inv_H_cache = []

        for p_idx, point in enumerate(self.points):
            yslice, xslice = self._padded_slice(point, self.roi_size, self.image_size, pad)
            F_padded = f[yslice, xslice].astype(np.float64)
            # F: unpadded subset of shape (h, w).
            F = F_padded[pad:-pad, pad:-pad]

            gx, gy = _gradient_dic(F, prefilter_gauss=self.prefilter_gauss)
            sd = _sd_images((gx, gy), jac)
            H = _hessian(sd)
            try:
                inv_H = np.linalg.inv(H)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    f"Degenerate ROI at point index {p_idx} (position {point}). "
                    f"The Hessian is singular (flat region or insufficient texture). "
                    f"Reposition this point away from uniform regions."
                ) from exc

            F_cache.append(F)
            F_mean_cache.append(F.mean())
            F_std_cache.append(F.std())
            sd_cache.append(sd)
            inv_H_cache.append(inv_H)

        self._F_cache = F_cache
        self._F_mean_cache = F_mean_cache
        self._F_std_cache = F_std_cache
        self._sd_cache = sd_cache
        self._inv_H_cache = inv_H_cache

    def show_points(self, figsize=(15, 5), cmap='gray', color='r'):
        """Show points to be analyzed, together with ROI borders.

        :param figsize: matplotlib figure size, defaults to ``(15, 5)``.
        :param cmap: matplotlib colormap, defaults to ``'gray'``.
        :param color: marker and border color, defaults to ``'r'``.
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.video.get_frame(0).astype(float), cmap=cmap)
        ax.scatter(self.points[:, 1],
                   self.points[:, 0], marker='.', color=color)

        for point in self.points:
            roi_border = patches.Rectangle(
                (point - self.roi_size // 2 - 0.5)[::-1],
                self.roi_size[1], self.roi_size[0],
                linewidth=1, edgecolor=color, facecolor='none',
            )
            ax.add_patch(roi_border)

        plt.grid(False)
        plt.show()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _jacobian_affine(h, w):
    """Affine warp Jacobian on centered subset coordinates.

    :param h: subset height.
    :type h: int
    :param w: subset width.
    :type w: int
    :return: array of shape ``(2, 6, h, w)``.
    :rtype: numpy.ndarray
    """
    yy, xx = np.meshgrid(
        np.arange(h) - h // 2,
        np.arange(w) - w // 2,
        indexing='ij',
    )
    xx = xx.astype(np.float64)
    yy = yy.astype(np.float64)
    zeros = np.zeros_like(xx)
    ones = np.ones_like(xx)

    jac = np.stack([
        np.stack([xx, yy, ones, zeros, zeros, zeros], axis=0),
        np.stack([zeros, zeros, zeros, xx, yy, ones], axis=0),
    ], axis=0)
    return jac


def _jacobian_rigid(h, w):
    """Rigid warp Jacobian (evaluated at ``p = 0``) on centered coords.

    :param h: subset height.
    :type h: int
    :param w: subset width.
    :type w: int
    :return: array of shape ``(2, 3, h, w)``.
    :rtype: numpy.ndarray
    """
    yy, xx = np.meshgrid(
        np.arange(h) - h // 2,
        np.arange(w) - w // 2,
        indexing='ij',
    )
    xx = xx.astype(np.float64)
    yy = yy.astype(np.float64)
    zeros = np.zeros_like(xx)
    ones = np.ones_like(xx)

    jac = np.stack([
        np.stack([ones, zeros, -yy], axis=0),
        np.stack([zeros, ones, xx], axis=0),
    ], axis=0)
    return jac


def _sd_images(grad, jac):
    """Compute the steepest-descent images.

    :param grad: tuple ``(gx, gy)`` of subset gradients with shape ``(h, w)``.
    :type grad: tuple of numpy.ndarray
    :param jac: warp Jacobian with shape ``(2, n_param, h, w)``.
    :type jac: numpy.ndarray
    :return: SD images with shape ``(n_param, h, w)``.
    :rtype: numpy.ndarray
    """
    gx, gy = grad
    return gx[None] * jac[0] + gy[None] * jac[1]


def _hessian(sd):
    """Compute the (Gauss-Newton) Hessian from the SD images.

    :param sd: SD images with shape ``(n_param, h, w)``.
    :type sd: numpy.ndarray
    :return: Hessian with shape ``(n_param, n_param)``.
    :rtype: numpy.ndarray
    """
    return np.einsum('ihw,jhw->ij', sd, sd)


def _affine_warp_matrix(p):
    """Build the 3x3 affine warp matrix from parameter vector ``p``.

    :param p: parameters ``[du/dx, du/dy, u, dv/dx, dv/dy, v]``.
    :type p: array_like of size 6
    :return: 3x3 warp matrix.
    :rtype: numpy.ndarray
    """
    return np.array([
        [1.0 + p[0], p[1], p[2]],
        [p[3], 1.0 + p[4], p[5]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _rigid_warp_matrix(p):
    """Build the 3x3 rigid warp matrix from parameter vector ``p``.

    :param p: parameters ``[u, v, phi]``.
    :type p: array_like of size 3
    :return: 3x3 warp matrix.
    :rtype: numpy.ndarray
    """
    c = np.cos(p[2])
    s = np.sin(p[2])
    return np.array([
        [c, -s, p[0]],
        [s, c, p[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _params_from_affine_matrix(M):
    """Extract the 6-component affine parameter vector from a 3x3 matrix.

    :param M: 3x3 affine warp matrix.
    :type M: numpy.ndarray
    :return: parameter vector ``[du/dx, du/dy, u, dv/dx, dv/dy, v]``.
    :rtype: numpy.ndarray
    """
    return np.array([
        M[0, 0] - 1.0, M[0, 1], M[0, 2],
        M[1, 0], M[1, 1] - 1.0, M[1, 2],
    ], dtype=np.float64)


def _params_from_rigid_matrix(M):
    """Extract the 3-component rigid parameter vector from a 3x3 matrix.

    :param M: 3x3 rigid warp matrix.
    :type M: numpy.ndarray
    :return: parameter vector ``[u, v, phi]``.
    :rtype: numpy.ndarray
    """
    phi = np.arctan2(M[1, 0], M[0, 0])
    return np.array([M[0, 2], M[1, 2], phi], dtype=np.float64)


def _gradient_dic(image, prefilter_gauss=True):
    """Compute image gradients using a 1-D finite-difference kernel.

    :param image: input 2-D image.
    :type image: numpy.ndarray
    :param prefilter_gauss: if True, use the Gauss-prefiltered kernel
        ``[-0.446, 0, 0.446]``. Otherwise use ``[-0.5, 0, 0.5]``.
    :type prefilter_gauss: bool
    :return: tuple ``(gx, gy)`` of arrays with the same shape as ``image``.
    :rtype: tuple of numpy.ndarray
    """
    if prefilter_gauss:
        k = np.array([-0.446, 0.0, 0.446], dtype=np.float64)
    else:
        k = np.array([-0.5, 0.0, 0.5], dtype=np.float64)

    kx = k[None, :]  # row kernel -> derivative along x (columns)
    ky = k[:, None]  # column kernel -> derivative along y (rows)

    gx = scipy.signal.convolve2d(image, kx, mode='same', boundary='symm')
    gy = scipy.signal.convolve2d(image, ky, mode='same', boundary='symm')
    return gx, gy


def _get_initial_guess(target_image, reference_subset):
    """Estimate an integer-pixel translation from FFT cross-correlation.

    :param target_image: full target image to search.
    :type target_image: numpy.ndarray
    :param reference_subset: reference subset to locate in ``target_image``.
    :type reference_subset: numpy.ndarray
    :return: integer translation ``(dy, dx)`` of the subset upper-left corner
        relative to its expected position (top-left ``(0, 0)``).
    :rtype: tuple of int
    """
    t = target_image - target_image.mean()
    r = reference_subset - reference_subset.mean()
    # Cross-correlation = convolution with the flipped template.
    corr = scipy.signal.fftconvolve(t, r[::-1, ::-1], mode='same')
    iy, ix = np.unravel_index(np.argmax(corr), corr.shape)
    cy = target_image.shape[0] // 2
    cx = target_image.shape[1] // 2
    return int(iy - cy), int(ix - cx)


# ---------------------------------------------------------------------------
# Multiprocessing
# ---------------------------------------------------------------------------


def multi(video: VideoReader, idi_method: 'DIC', processes, configuration_keys: list):
    """Split the points across processes and run the DIC method in parallel.

    :param video: VideoReader object.
    :type video: VideoReader
    :param idi_method: DIC instance to clone for each worker.
    :type idi_method: DIC
    :param processes: number of processes to run.
    :type processes: int
    :param configuration_keys: list of configuration keys forwarded to workers.
    :type configuration_keys: list
    :return: tuple ``(displacements, warp_params)`` concatenated across workers.
    :rtype: tuple of numpy.ndarray
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    if processes < 0:
        processes = cpu_count() + processes
    elif processes == 0:
        raise ValueError('Number of processes must not be zero.')

    points = idi_method.points
    points_split = tools.split_points(points, processes=processes)

    idi_kwargs = {
        'input_file': video.input_file,
        'root': video.root,
    }

    if video.file_format == 'np.ndarray':
        idi_kwargs['input_file'] = video.get_frames()

    exclude_keys = ["processes"]
    method_kwargs = dict(
        [(k, idi_method.__dict__.get(k, None)) for k in configuration_keys if k not in exclude_keys]
    )

    print(f'Computation start: {datetime.datetime.now()}')

    t_start = time.time()

    with rich_progress_bar_setup() as progress:
        futures = []
        with multiprocessing.Manager() as manager:
            _progress = manager.dict()

            with ProcessPoolExecutor(max_workers=processes) as executor:
                for n in range(0, len(points_split)):
                    task_id = progress.add_task(f"task {n} ({len(points_split[n])} points)")
                    futures.append(executor.submit(
                        worker, points_split[n], idi_kwargs, method_kwargs, n, _progress, task_id
                    ))

                while sum([future.done() for future in futures]) < len(futures):
                    for task_id, update_data in _progress.items():
                        latest = update_data["progress"]
                        total = update_data["total"]
                        progress.update(task_id, completed=latest, total=total + 1)

                out = []
                for future in futures:
                    out.append(future.result())

                out1 = sorted(out, key=lambda x: x[2])
                displacements_out = np.concatenate([d[0] for d in out1])
                warp_params_out = np.concatenate([d[1] for d in out1])

    t = time.time() - t_start
    minutes = t // 60
    seconds = t % 60
    hours = minutes // 60
    minutes = minutes % 60
    print(f'Computation duration: {hours:0>2.0f}:{minutes:0>2.0f}:{seconds:.2f}')

    return displacements_out, warp_params_out


def worker(points, idi_kwargs, method_kwargs, i, progress, task_id):
    """Worker function executed in each process.

    :return: tuple ``(displacements, warp_params, i)`` for reduction.
    :rtype: tuple
    """
    method_kwargs['show_pbar'] = False

    video = VideoReader(**idi_kwargs)
    idi = DIC(video)
    idi.configure(**method_kwargs)
    idi.configure_multiprocessing(i + 1, progress, task_id)
    idi.set_points(points)
    idi.get_displacements(autosave=False)
    warp_params = getattr(idi, 'warp_params', None)
    return idi.displacements, warp_params, i
