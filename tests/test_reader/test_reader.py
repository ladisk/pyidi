import numpy as np
from numpy.testing import assert_array_equal
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../../')

import pyidi

def test_1():
    video = pyidi.VideoReader(input_file='./data/data_synthetic_img_0.png')

    assert video.file_format == 'png'
    assert hasattr(video, "N")
    assert hasattr(video, "image_width")
    assert hasattr(video, "image_height")
    assert hasattr(video, "fps")

def test_get_frames():
    video = pyidi.VideoReader(input_file='./data/data_synthetic_img_0.png')

    assert video.get_frames().shape[0] == 10
    assert video.get_frames(4).shape[0] == 4
    assert video.get_frames((1, 5)).shape[0] == 4

def test_get_frames_mraw():
    video = pyidi.VideoReader(input_file='./data/data_showcase.cih')

    assert video.get_frames().shape[0] == 75
    assert video.get_frames(4).shape[0] == 4
    assert video.get_frames((1, 5)).shape[0] == 4
