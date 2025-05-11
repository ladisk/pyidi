import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyidi

def test_instance():
    video = pyidi.VideoReader(input_file='./data/data_synthetic.cih')
    lk = pyidi.LucasKanade(video)
    sof = pyidi.SimplifiedOpticalFlow(video)

    assert True
<<<<<<< HEAD
    # print('test_instance: passed')
=======
    print('test_instance: passed')
>>>>>>> e8ac912d11d741863a46428a4d83eb0fdc76a4d3
    return None

def test_points_sof():
    video = pyidi.VideoReader(input_file='./data/data_synthetic.cih')
    sof = pyidi.SimplifiedOpticalFlow(video)
    sof.set_points(points=[(0, 1), (1, 1)])

    assert True
<<<<<<< HEAD
    # print('test_points_sof: passed')
=======
    print('test_points_sof: passed')
>>>>>>> e8ac912d11d741863a46428a4d83eb0fdc76a4d3
    return None

def test_points_lk():
    video = pyidi.VideoReader(input_file='./data/data_synthetic.cih')
    lk = pyidi.LucasKanade(video)
    lk.set_points(points=[(0, 1), (1, 1)])

    assert True
<<<<<<< HEAD
    # print('test_points_lk: passed')
=======
    print('test_points_lk: passed')
>>>>>>> e8ac912d11d741863a46428a4d83eb0fdc76a4d3
    return None

# def test_info():
#     video = pyidi.pyIDI(input_file='./data/data_synthetic.cih')
#     assert 'Shutter Speed(s)' in video.reader.info.keys()
#     assert 'Color Bit' in video.reader.info.keys()
#     assert 'Total Frame' in video.reader.info.keys()
#     assert 'Record Rate(fps)' in video.reader.info.keys()


if __name__ == '__main__':
    test_instance()
    test_points_sof()
    test_points_lk()

# if __name__ == '__mains__':
#     np.testing.run_module_suite()