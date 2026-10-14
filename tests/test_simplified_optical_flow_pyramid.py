"""Tests for the pyramidal simplified optical flow method."""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift

import pyidi


def striped_video(displacements, root, height=200, width=200, seed=1):
    """Build a video of vertical stripes translated in the x direction.

    The pattern varies along x only, so the image gradient of every point is
    purely horizontal and the simplified optical flow can recover the whole
    displacement from a single point.

    :param displacements: x displacement of each frame, in pixels
    :type displacements: array_like
    :param root: directory the VideoReader writes its analysis folders to
    :type root: str
    :return: video reader over the generated frames
    :rtype: pyidi.VideoReader
    """
    rng = np.random.default_rng(seed)
    profile = gaussian_filter(rng.random(width), 4)
    profile = 255 * (profile - profile.min()) / np.ptp(profile)
    base = np.tile(profile, (height, 1))

    frames = np.array(
        [shift(base, [0, d], order=3, mode='nearest') for d in displacements])
    return pyidi.VideoReader(frames, root=str(root))


def test_subset_size_must_be_odd(tmp_path):
    video = striped_video([0., 0.5], tmp_path)
    method = pyidi.SimplifiedOpticalFlowPyramid(video)

    with pytest.raises(ValueError, match='odd'):
        method.configure(subset_size=4)
    with pytest.raises(ValueError, match='odd'):
        method.configure(subset_size=(3, 8))


def test_rectangular_subset_is_accepted(tmp_path):
    video = striped_video([0., 0.5], tmp_path)
    method = pyidi.SimplifiedOpticalFlowPyramid(video)
    method.configure(subset_size=(3, 9), reference_range=(0, 1),
                     progress_bar=False)

    assert method._subset_shape == (3, 9)


def test_levels_are_ordered_coarse_to_fine(tmp_path):
    video = striped_video([0., 0.5], tmp_path)
    method = pyidi.SimplifiedOpticalFlowPyramid(video)

    method.configure(levels=[0, 2, 1], reference_range=(0, 1),
                     progress_bar=False)
    assert method._levels == [2, 1, 0]

    method.configure(levels=None, n_levels=3, reference_range=(0, 1),
                     progress_bar=False)
    assert method._levels == [2, 1, 0]


def test_single_level_matches_subpixel_displacement(tmp_path):
    true_x = np.array([0., 0.2, -0.3, 0.4])
    video = striped_video(true_x, tmp_path)

    method = pyidi.SimplifiedOpticalFlowPyramid(video)
    method.configure(subset_size=3, levels=[0], reference_range=(0, 1),
                     progress_bar=False)
    method.set_points([[100, 60], [100, 100], [100, 140]])
    displacements = method.get_displacements(autosave=False)

    assert displacements.shape == (3, len(true_x), 2)
    np.testing.assert_allclose(
        displacements[:, :, 1].mean(axis=0), true_x, atol=0.1)
    np.testing.assert_allclose(displacements[:, :, 0], 0., atol=1e-8)


def test_pyramid_extends_the_displacement_range(tmp_path):
    # A single level only resolves displacements of about half a pixel. The
    # pyramid pre-shifts the subset, so several pixels stay in range.
    true_x = np.array([0., 3.2, 6.7, -5.1])
    video = striped_video(true_x, tmp_path)
    points = [[100, 60], [100, 100], [100, 140]]

    def estimate(levels):
        method = pyidi.SimplifiedOpticalFlowPyramid(video)
        method.configure(subset_size=(3, 9), levels=levels, sigma=1.5,
                         reference_range=(0, 1), progress_bar=False)
        method.set_points(points)
        return method.get_displacements(autosave=False)[:, :, 1].mean(axis=0)

    single_level = np.abs(estimate([0]) - true_x).max()
    pyramid = np.abs(estimate([3, 2, 1, 0]) - true_x).max()

    assert pyramid < 0.3
    assert pyramid < single_level


def test_points_outside_the_image_give_nan(tmp_path):
    video = striped_video([0., 0.3], tmp_path)

    method = pyidi.SimplifiedOpticalFlowPyramid(video)
    method.configure(subset_size=3, levels=[1, 0], reference_range=(0, 1),
                     progress_bar=False)
    method.set_points([[100, 100], [100, 500]])

    with pytest.warns(UserWarning, match='outside the image'):
        displacements = method.get_displacements(autosave=False)

    assert np.all(np.isnan(displacements[1]))
    assert np.all(np.isfinite(displacements[0]))
    np.testing.assert_array_equal(method.out_of_range_points, [1])


def test_convert_from_px_and_zero_shift(tmp_path):
    true_x = np.array([0., 0.3, -0.2, 0.4])
    video = striped_video(true_x, tmp_path)
    points = [[100, 100]]

    def estimate(**kwargs):
        method = pyidi.SimplifiedOpticalFlowPyramid(video)
        method.configure(subset_size=3, levels=[0], reference_range=(0, 1),
                         progress_bar=False, **kwargs)
        method.set_points(points)
        return method.get_displacements(autosave=False)

    plain = estimate()
    np.testing.assert_allclose(estimate(convert_from_px=2.), 2 * plain)
    np.testing.assert_allclose(
        estimate(zero_shift=True), plain - plain.mean(axis=1)[:, None, :],
        atol=1e-12)


def test_frame_range_limits_the_result(tmp_path):
    video = striped_video([0., 0.2, 0.4, 0.1, -0.3], tmp_path)

    method = pyidi.SimplifiedOpticalFlowPyramid(video)
    method.configure(subset_size=3, levels=[0], reference_range=(0, 1),
                     frame_range=(1, 4), progress_bar=False)
    method.set_points([[100, 100]])

    assert method.get_displacements(autosave=False).shape == (1, 3, 2)


def test_method_is_registered(tmp_path):
    # ``load_analysis`` maps the ``method`` field of settings.json through
    # ``method_mappings``, so a new method has to be registered there.
    import json

    from pyidi.load_analysis import method_mappings

    assert method_mappings['SimplifiedOpticalFlowPyramid'] is \
        pyidi.SimplifiedOpticalFlowPyramid

    video = striped_video([0., 0.3, -0.2], tmp_path)
    method = pyidi.SimplifiedOpticalFlowPyramid(video)
    method.configure(subset_size=(3, 5), levels=[1, 0], reference_range=(0, 1),
                     progress_bar=False)
    method.set_points([[100, 100], [100, 60]])
    method.get_displacements()

    with open(f'{method.root_this_analysis}/settings.json') as f:
        settings = json.load(f)

    assert settings['method'] == 'SimplifiedOpticalFlowPyramid'
    assert settings['settings']['subset_size'] == [3, 5]
    assert settings['settings']['levels'] == [1, 0]


@pytest.mark.parametrize('order', [0, 1, 3])
def test_subset_mean_matches_filtered_image(tmp_path, order):
    # Away from the edges, the subset mean read from the integral image equals
    # the subset-averaged image, interpolated with the same spline order.
    from scipy.ndimage import map_coordinates, uniform_filter

    video = striped_video([0., 0.3], tmp_path)
    method = pyidi.SimplifiedOpticalFlowPyramid(video)
    method.configure(subset_size=(5, 9), levels=[0], reference_range=(0, 1),
                     progress_bar=False)

    rng = np.random.default_rng(3)
    image = 1e4 + 1e3 * rng.random((80, 90))
    rows = rng.uniform(20, 60, 50)
    cols = rng.uniform(20, 70, 50)
    if order == 0:
        rows, cols = np.round(rows), np.round(cols)

    averaged = uniform_filter(image, size=(5, 9))
    expected = map_coordinates(averaged, [rows, cols], order=max(order, 1))

    np.testing.assert_allclose(
        method._subset_mean(method._integral_image(image), rows, cols, order),
        expected, rtol=0, atol=1e-8)


def test_subset_mean_is_cut_at_the_image_edge(tmp_path):
    video = striped_video([0., 0.3], tmp_path)
    method = pyidi.SimplifiedOpticalFlowPyramid(video)
    method.configure(subset_size=5, levels=[0], reference_range=(0, 1),
                     progress_bar=False)

    image = np.arange(100.).reshape(10, 10)
    mean = method._subset_mean(
        method._integral_image(image), np.array([0]), np.array([9]))

    np.testing.assert_allclose(mean, image[:3, 7:].mean())


def test_coarse_levels_keep_the_subset_on_the_point(tmp_path):
    # Dark lines on a light ground, 4 px of motion. A point at an odd column
    # falls between two pixels of level 1. Rounding it there moves the level 1
    # subset half a pixel off the line edge, and the pre-shift it hands to
    # level 0 comes out half a pixel short.
    index = np.arange(200) + 20
    profile = gaussian_filter(np.where(index % 40 < 6, 20., 235.), 1.2)
    base = np.tile(profile, (100, 1))
    frames = np.array([shift(base, [0, d], order=3, mode='nearest')
                       for d in (0., -4.)])
    video = pyidi.VideoReader(frames, root=str(tmp_path))

    method = pyidi.SimplifiedOpticalFlowPyramid(video)
    method.configure(subset_size=5, levels=[1, 0], interpolation='cubic',
                     reference_range=(0, 1), progress_bar=False)
    method.set_points([[50, 59], [50, 66]])   # rising and falling edge
    displacements = method.get_displacements(autosave=False)

    np.testing.assert_allclose(displacements[:, 1, 1], -4., atol=0.05)
    np.testing.assert_allclose(displacements[:, :, 0], 0., atol=1e-8)
