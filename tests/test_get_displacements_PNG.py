import numpy as np
from numpy.testing import assert_array_equal
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyidi

def test_multiprocessing():
    video = pyidi.pyIDI(input_file='./data/data_synthetic_img_0.png')
    video.set_method(method='lk', int_order=1, roi_size=(9, 9))

    points = np.array([
        [ 31,  35],
        [ 31, 215],
        [ 31, 126],
        [ 95,  71],
    ])
    video.set_points(points)
    video.method.configure(pbar_type='tqdm', multi_type='multiprocessing')
    res_1 = video.get_displacements(processes=2, resume_analysis=False, autosave=False)

    video.method.configure(pbar_type='atpbar', multi_type='mantichora')
    res_2 = video.get_displacements(processes=2, resume_analysis=False, autosave=False)

    assert_array_equal(res_1, res_2)

if __name__ == '__main__':
    test_multiprocessing()