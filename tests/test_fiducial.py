import numpy as np
import sys, os
from sdypy.io import sfmov
import cv2
import pytest

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyidi

def test_instance():
    """
    Test creation of Fiducial instance with valid video data.
    Checks that the loaded video is a NumPy array with correct dimensions,
    and verifies internal conversion to grayscale if needed.
    Also tests that invalid input shapes raise the expected error.
    """
    video_path = os.path.join(my_path, '..', 'data', 'data_fiducial.sfmov')
    
    # Load the video (grayscale or RGB)
    data = sfmov.get_data(video_path)

    # Verify loaded data type and dimensions
    assert isinstance(data, np.ndarray), "Loaded data is not a NumPy array"
    assert data.ndim in [3, 4], f"Expected 3D or 4D array, got {data.ndim}D"

    # Instantiate Fiducial and check attributes
    test = pyidi.fiducial.Fiducial(data)
    assert isinstance(test, pyidi.fiducial.Fiducial), "Failed to create Fiducial instance"
    assert hasattr(test, 'video'), "Fiducial object missing 'video' attribute"

    # Check that internal video is grayscale with matching frame count
    assert test.video.ndim == 3, f"Expected grayscale 3D video, got {test.video.ndim}D"
    assert test.video.shape[0] == data.shape[0], "Frame count mismatch after conversion"
    
    # Test that invalid input shapes raise ValueError
    invalid_data = np.random.rand(64, 64)
    with pytest.raises(ValueError):
        pyidi.fiducial.Fiducial(invalid_data)


def test_instance_rgb2gray():
    """
    Test that an RGB video input is correctly converted to grayscale internally.
    """
    # Create dummy RGB video: 10 frames, 64x64 pixels, 3 color channels
    rgb_data = np.random.randint(0, 255, (10, 64, 64, 3), dtype=np.uint8)
    
    # Instantiate Fiducial and check internal video shape
    test = pyidi.fiducial.Fiducial(rgb_data)
    assert test.video.shape == (10, 64, 64), "RGB to grayscale conversion failed"


def test_compensation():
    """
    Test that the uncertainty (transformation error) at the reference frame after
    fiducial-based compensation is effectively zero.
    """
    video_path = os.path.join(my_path, '..', 'data', 'data_fiducial.sfmov')
    data = sfmov.get_data(video_path)
    test = pyidi.fiducial.Fiducial(data)
    
    # Pre-process video for better marker detection
    processed = test.pre_process(clip_range=(19.4, 20.0), enhance_contrast=True)
    
    # Detect fiducial markers in processed frames
    fiducials = test.detect_markers(processed)
    
    # Randomly select a reference frame index
    ref = np.random.randint(0, data.shape[0] - 1)
    
    # Compute transformation matrices relative to the reference frame
    matrices = test.compute_transformations(fiducials, reference_index=ref)
    
    # Revert fiducial coordinates using the computed transformations
    fiducial_reverted = test.revert_fiducial(fiducials, matrices)
    
    # Analyze uncertainty (mean error) between original and reverted fiducials
    stats = test.uncertainty_analysis(fiducials, fiducial_reverted)

    # Assert that the error at the reference frame is negligible (within tolerance)
    assert abs(stats['Per-frame Mean Error'][ref]) < 1e-3


def test_revert_frames_shape():
    """
    Test that the video frames reverted (compensated) using transformation matrices
    have the same shape as the original video.
    """
    video_path = os.path.join(my_path, '..', 'data', 'data_fiducial.sfmov')
    data = sfmov.get_data(video_path)
    test = pyidi.fiducial.Fiducial(data)
    
    # Pre-process video and detect fiducials
    processed = test.pre_process(clip_range=(19.4, 20.0), enhance_contrast=True)
    fiducials = test.detect_markers(processed)
    
    # Compute transformations
    matrices = test.compute_transformations(fiducials)
    
    # Apply compensation to video frames
    compensated = test.revert_frames(matrices)
    
    # Verify that compensated video shape matches original video shape
    assert compensated.shape == test.video.shape, "Compensated video shape mismatch"


if __name__ == '__main__':
    test_instance()
    test_instance_rgb2gray()
    test_compensation()
    test_revert_frames_shape()
