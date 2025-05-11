import sys, os
import pyMRAW
import numpy as np

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../../')

import pyidi

def test_png_stream():
    video = pyidi.VideoReader(input_file='./data/data_synthetic_img_0.png')

    assert video.file_format == 'png'
    assert hasattr(video, "N")
    assert hasattr(video, "image_width")
    assert hasattr(video, "image_height")
    assert hasattr(video, "fps")
    # print('test_png_stream: passed')
    return None

def test_mp4():
    video = pyidi.VideoReader(input_file='./data/data_synthetic.mp4')

    assert video.file_format == 'mp4'
    assert hasattr(video, "N")
    assert hasattr(video, "image_width")
    assert hasattr(video, "image_height")
    assert hasattr(video, "fps")
    # print('test_mp4: passed')
    return None

def test_ndarray():
    data, info = pyMRAW.load_video('./data/data_showcase.cih')
    video = pyidi.VideoReader(data, root='./data/', fps=info.get('Record Rate(fps)'))

    assert video.file_format == 'np.ndarray'
    assert hasattr(video, "N")
    assert hasattr(video, "image_width")
    assert hasattr(video, "image_height")
    assert hasattr(video, "fps")
    # print('test_ndarray: passed')
    return None

def test_get_frames():
    video = pyidi.VideoReader(input_file='./data/data_synthetic_img_0.png')

    assert video.get_frames().shape[0] == 10
    assert video.get_frames(4).shape[0] == 4
    assert video.get_frames((1, 5)).shape[0] == 4
    # print('test_get_frames: passed')
    return None

def test_get_frames_mraw():
    video = pyidi.VideoReader(input_file='./data/data_showcase.cih')

    assert video.get_frames().shape[0] == 75
    assert video.get_frames(4).shape[0] == 4
    assert video.get_frames((1, 5)).shape[0] == 4
    # print('test_get_frames_mraw: passed')
    return None

def test_get_frames_ndarray():
    data, info = pyMRAW.load_video('./data/data_showcase.cih')
    video = pyidi.VideoReader(data, root='./data/', fps=info.get('Record Rate(fps)'))

    assert video.get_frames().shape[0] == 75
    assert video.get_frames(4).shape[0] == 4
    assert video.get_frames((1, 5)).shape[0] == 4
    # print('test_get_frames_ndarray: passed')
    return None


def test_get_frames_mp4():
    video = pyidi.VideoReader(input_file='./data/data_synthetic.mp4')

    assert video.get_frames().shape[0] == 10
    assert video.get_frames(use_channel='R').shape[0] == 10
    assert video.get_frames(use_channel='G').shape[0] == 10
    assert video.get_frames(use_channel='B').shape[0] == 10
    assert video.get_frames(4).shape[0] == 4
    assert video.get_frames((1, 5)).shape[0] == 4
    # print('test_get_frames_mp4: passed')
    return None

if __name__ == '__main__':
    test_png_stream()
    test_mp4()
    test_ndarray()
    test_get_frames()
    test_get_frames_mraw()
    test_get_frames_ndarray()
    test_get_frames_mp4()