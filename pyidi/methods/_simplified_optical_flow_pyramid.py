import warnings

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from tqdm import tqdm

try:
    from qtpy.QtWidgets import QApplication
except ImportError:
    QApplication = None

from .idi_method import IDIMethod

# Spline order each interpolation name maps to. 'nearest' does no
# interpolation at all, so it has no entry.
INTERPOLATION_ORDERS = {'linear': 1, 'quadratic': 2, 'cubic': 3}


class SimplifiedOpticalFlowPyramid(IDIMethod):
    """
    Displacement computation based on the simplified optical flow method [1],
    evaluated on a coarse-to-fine image pyramid.

    Each point is treated on its own. In a rectangular subset around the point,
    the method averages the reference intensity and the two image gradients. The
    intensity difference between the current and the reference frame gives the
    displacement along the gradient direction, which is then projected onto the
    image axes through the gradient components:

    .. math::

        \\mathbf{d} = \\frac{I_\\mathrm{ref} - I_t}{|\\nabla I|^2}\\, \\nabla I

    The simplified optical flow assumes displacements below half a pixel. The
    pyramid lifts that limit: the estimate from a coarser level is scaled up and
    applied as a pre-shift of the subset on the next level, so only the
    sub-pixel residual is measured at full resolution. The pre-shift is rounded
    to whole pixels with ``'nearest'`` interpolation and stays fractional with
    the spline interpolations.

    Literature:
        [1] Javh, J., Slavič, J., & Boltežar, M. (2017). The subpixel resolution
            of optical-flow-based modal analysis. Mechanical Systems
            and Signal Processing, 88, 89-99.
        [2] Bouguet, J.-Y. (2001). Pyramidal implementation of the affine Lucas
            Kanade feature tracker. Intel Corporation, Microprocessor Research
            Labs.
    """

    def configure(self, subset_size=3, levels=None, n_levels=2, scale=2,
                  sigma=1., interpolation='nearest', reference_range=(0, 100),
                  frame_range='all', convert_from_px=1., zero_shift=False,
                  progress_bar=True):
        """
        Set the attributes and build the reference pyramid.

        :param subset_size: size of the averaging subset around each point. Pass an
            int for a square subset, or a ``(size_y, size_x)`` pair for a rectangular
            one. Both sizes must be odd. Defaults to 3.
        :type subset_size: int or tuple of int, optional
        :param levels: pyramid levels to use, from any order, for example ``[0, 1, 2]``
            or ``[0, 2]``. Level 0 is full resolution and a higher number is a more
            reduced image. Intermediate levels can be skipped. If None, ``n_levels``
            is used. Defaults to None.
        :type levels: list of int or None, optional
        :param n_levels: number of pyramid levels to use when ``levels`` is not given,
            that is, levels 0 to ``n_levels - 1``. Defaults to 2.
        :type n_levels: int, optional
        :param scale: down-sampling factor between two neighbouring levels,
            defaults to 2
        :type scale: int, optional
        :param sigma: standard deviation, in pixels, of the Gaussian blur applied
            before down-sampling, defaults to 1.
        :type sigma: float, optional
        :param interpolation: how the estimate from a coarser level shifts the
            subset on the next one. ``'nearest'`` rounds the shift to whole
            pixels, which is fast but leaves a step in the result wherever the
            rounding changes. ``'linear'``, ``'quadratic'`` and ``'cubic'``
            keep the shift fractional and read the subset with a spline of
            order 1, 2 or 3. The
            cubic spline smooths the image far less than the linear one, so it
            carries much less bias. Defaults to ``'nearest'``.
        :type interpolation: str, optional
        :param reference_range: range of frames averaged into the reference image,
            defaults to (0, 100)
        :type reference_range: tuple, optional
        :param frame_range: range of frames to compute the displacements for, or
            'all', defaults to 'all'
        :type frame_range: str or tuple, optional
        :param convert_from_px: distance unit per pixel, defaults to 1.
        :type convert_from_px: float, optional
        :param zero_shift: shift the mean of each displacement signal to zero,
            defaults to False
        :type zero_shift: bool, optional
        :param progress_bar: show a progress bar while computing, defaults to True
        :type progress_bar: bool, optional
        """
        if subset_size is not None:
            self.subset_size = subset_size
        if levels is not None:
            self.levels = levels
        elif not hasattr(self, 'levels'):
            self.levels = None
        if n_levels is not None:
            self.n_levels = n_levels
        if scale is not None:
            self.scale = scale
        if sigma is not None:
            self.sigma = sigma
        if interpolation is not None:
            self.interpolation = interpolation
        if reference_range is not None:
            self.reference_range = reference_range
        if frame_range is not None:
            self.frame_range = frame_range
        if convert_from_px is not None:
            self.convert_from_px = convert_from_px
        if zero_shift is not None:
            self.zero_shift = zero_shift
        if progress_bar is not None:
            self.progress_bar = progress_bar

        # No local variables here: ``get_displacements`` reads
        # ``configure.__code__.co_varnames``, which lists locals as well as
        # arguments, and passes every name back in as a keyword argument.
        if self.interpolation not in ('nearest', *INTERPOLATION_ORDERS):
            raise ValueError(
                "interpolation must be one of 'nearest', "
                + "', '".join(INTERPOLATION_ORDERS)
                + f', got {self.interpolation!r}.')

        self._subset_shape = self._validate_subset_size(self.subset_size)
        self._levels = self._validate_levels(self.levels, self.n_levels)

        self._reference()

    def _validate_subset_size(self, subset_size):
        """Turn ``subset_size`` into a ``(size_y, size_x)`` pair of odd integers.

        :param subset_size: subset size as configured
        :type subset_size: int or tuple of int
        :return: subset size in the y and x direction
        :rtype: tuple of int
        """
        if np.ndim(subset_size) == 0:
            shape = (int(subset_size), int(subset_size))
        else:
            shape = tuple(int(_) for _ in subset_size)

        if len(shape) != 2:
            raise ValueError(
                f'subset_size must be an int or a (size_y, size_x) pair, got {subset_size}.')
        if any(_ < 1 or _ % 2 == 0 for _ in shape):
            raise ValueError(
                f'Both subset sizes must be odd and positive, got {shape}.')
        return shape

    def _validate_levels(self, levels, n_levels):
        """Sort the requested pyramid levels from the coarsest to the finest.

        :param levels: levels to use, or None to use levels 0 to ``n_levels - 1``
        :type levels: list of int or None
        :param n_levels: number of levels to use when ``levels`` is None
        :type n_levels: int
        :return: levels, coarsest first
        :rtype: list of int
        """
        if levels is None:
            levels = range(int(n_levels))

        levels = sorted({int(_) for _ in levels}, reverse=True)
        if not levels or levels[-1] < 0:
            raise ValueError(
                'levels must be a non-empty list of non-negative levels, '
                'for example [0, 1, 2] or [0, 2].')
        return levels

    def _down_sample(self, image):
        """Blur an image and take every ``scale``-th pixel.

        :param image: image to reduce
        :type image: numpy.ndarray
        :return: image reduced by the factor ``scale``
        :rtype: numpy.ndarray
        """
        return gaussian_filter(image, self.sigma)[::self.scale, ::self.scale]

    def _reference(self):
        """Average the reference frames and build the reference pyramid.

        For every used level the method stores the integral images of the
        reference image and of its two gradients, from which the subset means
        around the points are read later.
        """
        reference_range = tuple(self.reference_range)
        if reference_range[1] >= self.video.N:
            reference_range = (reference_range[0], self.video.N)
        self.reference_range = reference_range

        reference_image = np.zeros(
            (self.video.image_height, self.video.image_width), dtype=float)
        for frame in range(reference_range[0], reference_range[1]):
            reference_image += self.video.get_frame(frame)
        reference_image /= (reference_range[1] - reference_range[0])

        self.reference_image = reference_image

        # Level 0 is the full resolution image, every further level is reduced
        # by ``scale``. The pyramid is built up to the coarsest used level, even
        # if the levels in between are not used.
        pyramid = [reference_image]
        for _ in range(self._levels[0]):
            pyramid.append(self._down_sample(pyramid[-1]))

        self._reference_levels = {}
        for level in self._levels:
            image = pyramid[level]
            gradient_0, gradient_1 = np.gradient(image)
            self._reference_levels[level] = {
                'image': self._integral_image(image),
                'gradient_0': self._integral_image(gradient_0),
                'gradient_1': self._integral_image(gradient_1),
                'shape': image.shape,
            }

    def _integral_image(self, data):
        """Build the integral image (summed-area table) of an image.

        Element ``[i, j]`` of the table holds the sum of ``data[:i, :j]``, so
        the table has one row and one column more than the image. The mean of
        the image is subtracted first and returned alongside the table: the
        sums then stay small, and the difference of four table values that
        gives a subset sum loses less precision.

        :param data: image to integrate
        :type data: numpy.ndarray
        :return: the table and the subtracted mean, under the keys
            ``'table'`` and ``'offset'``
        :rtype: dict
        """
        offset = np.mean(data, dtype=float)
        table = np.zeros((data.shape[0] + 1, data.shape[1] + 1))
        inner = table[1:, 1:]
        np.subtract(data, offset, out=inner)
        np.cumsum(inner, axis=0, out=inner)
        np.cumsum(inner, axis=1, out=inner)
        return {'table': table, 'offset': offset}

    def _subset_mean(self, integral, rows, cols, order=0):
        """Read the mean of the subset centred on each point from an integral
        image.

        The subset spans the pixels from ``row - size_y // 2`` to
        ``row + size_y // 2``, and likewise in x. A subset that reaches over
        the image edge is cut to the pixels inside the image.

        With a fractional centre, the subset edges fall between pixels. The
        integral image is then interpolated at the four corners. A bilinear
        interpolation (``order=1``) weighs the partly covered edge pixels by
        the covered fraction, which equals linear interpolation between the
        means of the neighbouring whole-pixel subsets. A quadratic or cubic
        spline (``order=2`` or ``3``) interpolates the integral image smoothly
        instead.

        :param integral: integral image from ``_integral_image``
        :type integral: dict
        :param rows: row of each subset centre
        :type rows: numpy.ndarray
        :param cols: column of each subset centre
        :type cols: numpy.ndarray
        :param order: 0 for whole-pixel centres, read without interpolation,
            or the spline order used at fractional centres, defaults to 0
        :type order: int, optional
        :return: subset mean around each point
        :rtype: numpy.ndarray
        """
        table = integral['table']
        height, width = table.shape[0] - 1, table.shape[1] - 1
        half_y, half_x = self._subset_shape[0] // 2, self._subset_shape[1] // 2

        top = np.clip(rows - half_y, 0, height)
        bottom = np.clip(rows + half_y + 1, 0, height)
        left = np.clip(cols - half_x, 0, width)
        right = np.clip(cols + half_x + 1, 0, width)

        corner_rows = np.concatenate([top, top, bottom, bottom])
        corner_cols = np.concatenate([left, right, left, right])
        if order == 0:
            corners = table[corner_rows.astype(int), corner_cols.astype(int)]
        else:
            # A single call, so the spline is fitted to the table only once.
            corners = map_coordinates(
                table, np.vstack([corner_rows, corner_cols]), order=order,
                mode='nearest')
        top_left, top_right, bottom_left, bottom_right = corners.reshape(4, -1)

        area = (bottom - top) * (right - left)
        return ((bottom_right - bottom_left - top_right + top_left) / area
                + integral['offset'])

    def _reference_at_points(self, level, points):
        """Read the reference subset means around the points of one level.

        :param level: pyramid level
        :type level: int
        On a coarse level a point generally falls between pixels. Its subset
        is then read with interpolation rather than moved to the nearest
        pixel, so that every level centres the subset on the same spot of the
        image. With ``'nearest'`` interpolation the spline order off the grid
        is 1.

        :param level: pyramid level
        :type level: int
        :param points: points in the pixel coordinates of this level, possibly
            fractional
        :type points: numpy.ndarray
        :return: the points, the level shape, the spline order to read the
            current frame with, and the subset means of the reference image,
            both gradients and the gradient magnitude
        :rtype: dict
        """
        reference = self._reference_levels[level]
        rows, cols = points[:, 0], points[:, 1]

        # Whole-pixel points need no interpolation for the reference, and
        # with 'nearest' neither for the current frame, whose shift is whole
        # pixels as well.
        on_grid = np.array_equal(points, np.round(points))
        order = INTERPOLATION_ORDERS.get(self.interpolation, 0 if on_grid else 1)

        at_points = {'points': points, 'shape': reference['shape'],
                     'order': order}
        for key in ('image', 'gradient_0', 'gradient_1'):
            at_points[key] = self._subset_mean(
                reference[key], rows, cols, order=0 if on_grid else order)
        at_points['gradient_magnitude'] = np.sqrt(
            at_points['gradient_0']**2 + at_points['gradient_1']**2)
        return at_points

    def _level_points(self, level):
        """Map the points to the pixel coordinates of a pyramid level.

        Pixel ``i`` of level ``k`` sits on pixel ``i * scale**k`` of level 0,
        so a point maps to ``point / scale**k``. The result is not rounded: a
        rounded point would move the subset off the point by up to half a
        pixel of the level, which on level 2 is two pixels of level 0.

        :param level: pyramid level
        :type level: int
        :return: points on that level, as floats, and their in-range mask
        :rtype: tuple of numpy.ndarray
        """
        points = np.asarray(self.points, dtype=float) / self.scale**level
        height, width = self._reference_levels[level]['shape']

        # A point within half a pixel of the image still belongs to its edge
        # pixel.
        valid = ((points[:, 0] >= -0.5) & (points[:, 0] < height - 0.5) &
                 (points[:, 1] >= -0.5) & (points[:, 1] < width - 0.5))
        points[:, 0] = np.clip(points[:, 0], 0, height - 1)
        points[:, 1] = np.clip(points[:, 1], 0, width - 1)
        return points, valid

    def calculate_displacements(self):
        """Compute the displacements of the set points, coarse to fine.

        The result is written to ``self.displacements`` with the shape
        ``(n_points, n_time_points, 2)``, in (y, x) and in pixels, unless
        ``convert_from_px`` says otherwise.
        """
        if not hasattr(self, 'points'):
            raise Exception('Please set points for analysis!')

        if self.frame_range != 'all':
            frames = range(self.frame_range[0], self.frame_range[1])
        else:
            frames = range(self.video.N)

        self.displacements = np.zeros((self.points.shape[0], len(frames), 2))

        # The points and the reference data do not change between frames, so
        # they are mapped onto every used level once.
        level_references = {}
        in_range = np.ones(self.points.shape[0], dtype=bool)
        for level in self._levels:
            points, valid = self._level_points(level)
            level_references[level] = self._reference_at_points(level, points)
            in_range &= valid

        if not np.all(in_range):
            warnings.warn(
                f'{np.count_nonzero(~in_range)} of {in_range.size} points fall '
                'outside the image on at least one pyramid level. Their '
                'displacements are set to NaN.')

        self.out_of_range_points = np.nonzero(~in_range)[0]

        p_bar = tqdm if self.progress_bar else (lambda x, **kwargs: x)

        for i, frame_number in enumerate(p_bar(frames, ncols=100)):
            image = self.video.get_frame(frame_number).astype(float)
            self.displacements[:, i, :] = self._displacement_in_frame(
                image, level_references)

            # Keep a Qt based GUI responsive while the analysis runs.
            if QApplication is not None and QApplication.instance() is not None:
                QApplication.processEvents()

        self.displacements[~in_range] = np.nan
        self.displacements *= self.convert_from_px

        if self.zero_shift:
            self.displacements -= np.nanmean(
                self.displacements, axis=1)[:, None, :]

    def _displacement_in_frame(self, image, level_references):
        """Estimate the displacement of every point in a single frame.

        :param image: current frame at full resolution
        :type image: numpy.ndarray
        :param level_references: points and reference subset means of each
            used level, from ``_reference_at_points``
        :type level_references: dict
        :return: displacement of each point in level 0 pixels, in (y, x)
        :rtype: numpy.ndarray
        """
        pyramid = [image]
        for _ in range(self._levels[0]):
            pyramid.append(self._down_sample(pyramid[-1]))

        displacement = np.zeros((self.points.shape[0], 2))
        previous_level = self._levels[0]

        for level in self._levels:
            # Carry the estimate from the previous level over to this one.
            displacement = displacement * self.scale**(previous_level - level)

            # The pre-shift moves the subset onto the pattern, so only the
            # small residual is left for the simplified optical flow to
            # measure. Whole pixels need no interpolation, hence the default.
            if self.interpolation == 'nearest':
                pre_shift = np.round(displacement)
            else:
                pre_shift = displacement
            residual = self._residual_displacement(
                pyramid[level], level_references[level], pre_shift)

            displacement = pre_shift + residual
            previous_level = level

        # The finest used level is not necessarily level 0.
        return displacement * self.scale**previous_level

    def _residual_displacement(self, image, reference, pre_shift):
        """Measure the sub-pixel displacement left after the pre-shift.

        :param image: current frame at this pyramid level
        :type image: numpy.ndarray
        :param reference: points and reference subset means of this level,
            from ``_reference_at_points``
        :type reference: dict
        :param pre_shift: subset shift of each point, in (y, x); whole pixels
            with ``'nearest'`` interpolation, fractional otherwise
        :type pre_shift: numpy.ndarray
        :return: residual displacement of each point, in (y, x)
        :rtype: numpy.ndarray
        """
        points = reference['points']
        height, width = reference['shape']

        rows = np.clip(points[:, 0] + pre_shift[:, 0], 0, height - 1)
        cols = np.clip(points[:, 1] + pre_shift[:, 1], 0, width - 1)

        # One pass over the image builds the integral image, after which each
        # subset mean costs four reads, whatever the subset size.
        current = self._subset_mean(
            self._integral_image(image), rows, cols, order=reference['order'])
        difference = reference['image'] - current

        # Displacement along the gradient direction, projected onto the image
        # axes through the gradient components.
        with np.errstate(divide='ignore', invalid='ignore'):
            along_gradient = difference / reference['gradient_magnitude']**2
        along_gradient[~np.isfinite(along_gradient)] = 0.

        return np.column_stack([along_gradient * reference['gradient_0'],
                                along_gradient * reference['gradient_1']])
