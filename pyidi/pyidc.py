import numpy as np
import collections

from ._simplified_optical_flow import *

class pyIDI:
    def __init__(self, cih_file):
        self.cih_file = cih_file

        self.avaliable_methods = {
            'simplified_optical_flow': SimplifiedOpticalFlow,
            'sof': SimplifiedOpticalFlow,
        }

        self.mraw, self.info = self.load_video()

    def set_points(self, points=None, method='simplified_optical_flow', **kwargs):
        """Set points that will be used to calculate displacements.
        
        The method of displacement calculation must be specified here.
        If `points` is None, a method is called to help the user determine the points.
        """
        if method in self.avaliable_methods.keys():
            self.method = self.avaliable_methods[method]
        else:
            self.method = method

        if points is None:
            self.method.get_points(self, **kwargs) # get_points sets the attribute video.points
        else:
            self.points = points

    def show_points(self):
        """Show selected points on image.
        """
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.imshow(self.mraw[0].astype(float), cmap='gray')
        ax.scatter(self.points[:, 1], self.points[:, 0], marker='.', color='r')
        plt.grid(False)
        plt.show()

    def get_displacements(self, **kwargs):
        """Calculate the displacements based on chosen method.
        """
        self.method_object = self.method(self, **kwargs)

        self.method_object.calculate_displacements(self)
        return self.method_object.displacements

    def load_video(self):
        """Get video and it's information.
        """
        info = self.get_CIH_info()
        self.N = int(info['Total Frame'])
        self.image_width = int(info['Image Width'])
        self.image_height = int(info['Image Height'])
        bit = info['Color Bit']

        if bit == '16':
            self.bit_dtype = np.uint16
        elif bit == '8':
            self.bit_dtype = np.uint8

        filename = '.'.join(self.cih_file.split('.')[:-1])
        mraw = np.memmap(filename+'.mraw', dtype=self.bit_dtype, mode='r', shape=(self.N, self.image_height, self.image_width))
        return mraw, info
    
    def get_CIH_info(self):
        """Get info from .cih file in path, return it as dict.

        https://github.com/ladisk/pyDIC/blob/master/py_dic/dic_tools.py - Domen Gorjup
        """
        wanted_info = ['Date',
                    'Camera Type',
                    'Record Rate(fps)',
                    'Shutter Speed(s)',
                    'Total Frame',
                    'Image Width',
                    'Image Height',
                    'File Format',
                    'EffectiveBit Depth',
                    'Comment Text',
                    'Color Bit']

        info_dict = collections.OrderedDict([])

        with open(self.cih_file, 'r') as file:
            for line in file:
                line = line.rstrip().split(' : ')
                if line[0] in wanted_info:                
                    key, value = line[0], line[1]#[:20]
                    info_dict[key] = bytes(value, "utf-8").decode("unicode_escape") # Evaluate escape characters

        return info_dict




    
