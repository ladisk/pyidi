import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyidi

def test_cih_lk():
    video = pyidi.VideoReader(input_file='./data/data_synthetic.cih')
    lk = pyidi.LucasKanade(video)

    points = np.array([
        [ 31,  35],
        [ 31, 215],
        [ 31, 126],
        [ 95,  71],
    ])
    lk.set_points(points)
    lk.configure(int_order=1, verbose=0, show_pbar=False)
    dsp = lk.get_displacements(autosave=False)

    np.testing.assert_array_equal(dsp.shape, (len(points), video.N, 2))
    # print('Displacements shape:', dsp.shape)

    return None

def test_cih_lk1d():
    video = pyidi.VideoReader(input_file='./data/data_synthetic.cih')
    lk = pyidi.DirectionalLucasKanade(video)

    points = np.array([
        [ 31,  35],
        [ 31, 215],
        [ 31, 126],
        [ 95,  71],
    ])
    lk.set_points(points)
    lk.configure(dij = (0.5, 0.5), int_order=1, verbose=0, show_pbar=False)
    dsp = lk.get_displacements(autosave=False)

    np.testing.assert_array_equal(dsp.shape, (len(points), video.N, 2))
    return None

def test_png_sof():
    video = pyidi.VideoReader(input_file='./data/data_synthetic_img_0.png')
    sof = pyidi.SimplifiedOpticalFlow(video)
    
    points = np.array([
        [ 31,  35],
        [ 31, 215],
        [ 31, 126],
        [ 95,  71],
    ])
    sof.set_points(points)
    dsp = sof.get_displacements(autosave=False, progress_bar=False)

    np.testing.assert_array_equal(dsp.shape, (len(points), video.N, 2))
    # print('Displacements shape:', dsp.shape)

    return None

if __name__ == '__main__':
    test_cih_lk()
    test_cih_lk1d()
    test_png_sof()