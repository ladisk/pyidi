import numpy as np
from numpy.testing import assert_array_equal
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyidi

def test():
    video = pyidi.pyIDI(input_file='./data/data_synthetic_img_0.png')
    video.set_method(method='lk')
    video.method.configure(int_order=1, roi_size=(9, 9))
    
    points = np.array([
        [ 31,  35],
        [ 31, 215],
        [ 31, 126],
        [ 95,  71],
    ])
    video.set_points(points)
    video.method.configure(show_pbar=False)
    res_1 = video.get_displacements(resume_analysis=False, autosave=False)

if __name__ == '__main__':
    test()