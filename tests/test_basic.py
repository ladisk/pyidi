import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyidi

def test_instance():
    video = pyidi.pyIDI(input_file='./data/data_synthetic.cih')
    video.set_method(method='sof')
    video.set_method(method='lk')
    assert True

def test_points():
    video = pyidi.pyIDI(input_file='./data/data_synthetic.cih')
    video.set_points(points=[(0, 1), (1, 1)])
    video.set_method(method='sof', reference_range=(0, 2))

def test_points_1():
    video = pyidi.pyIDI(input_file='./data/data_synthetic.cih')
    video.set_method(method='sof')
    video.set_points(points=[(0, 1), (1, 1)])

# def test_info():
#     video = pyidi.pyIDI(input_file='./data/data_synthetic.cih')
#     assert 'Shutter Speed(s)' in video.reader.info.keys()
#     assert 'Color Bit' in video.reader.info.keys()
#     assert 'Total Frame' in video.reader.info.keys()
#     assert 'Record Rate(fps)' in video.reader.info.keys()


# if __name__ == '__main__':
#     test_info()

if __name__ == '__mains__':
    np.testing.run_module_suite()