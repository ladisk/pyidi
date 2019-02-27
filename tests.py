import unittest

import os
import sys
sys.path.insert(0, os.path.realpath('__file__'))
import pyidi


class TestPyIDI(unittest.TestCase):

    def test_instance(self):
        video = pyidi.pyIDI(cih_file='./data/data_showcase.cih')
        video.set_method(method='sof')
        video.set_method(method='gb')
    
    def test_points(self):
        video = pyidi.pyIDI(cih_file='./data/data_showcase.cih')
        video.set_points(points=[(0, 1), (1, 1)])
        video.set_method(method='sof')

    def test_points_1(self):
        video = pyidi.pyIDI(cih_file='./data/data_showcase.cih')
        video.set_method(method='sof')
        video.set_points(points=[(0, 1), (1, 1)])

    def test_info(self):
        video = pyidi.pyIDI(cih_file='./data/data_showcase.cih')
        self.assertIn('Shutter Speed(s)', video.info.keys())
        self.assertIn('Color Bit', video.info.keys())
        self.assertIn('Total Frame', video.info.keys())
        self.assertIn('Record Rate(fps)', video.info.keys())


if __name__ == '__main__':
    unittest.main()