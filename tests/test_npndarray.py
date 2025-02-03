import numpy as np
import pyMRAW
from numpy.testing import assert_array_equal
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyidi

def test():
    # data = np.load(r".\data\data_synthetic_generated.npy")
    data, _ = pyMRAW.load_video('./data/data_synthetic.cih')
    data = np.array(data.tolist())


    video = pyidi.pyIDI(input_file=data, root='./data')
    video.set_method(method='lk')
    video.method.configure(int_order=1, roi_size=(9, 9), show_pbar=False)

    points = np.array([
        [ 31,  35],
        [ 31, 215],
        [ 31, 126],
        [ 95,  71],
    ])

    video.set_points(points)
    res_1 = video.get_displacements(resume_analysis=False, autosave=False)


if __name__ == '__main__':
    test()