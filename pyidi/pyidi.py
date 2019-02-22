import numpy as np
import collections
import matplotlib.pyplot as plt

from .methods import SimplifiedOpticalFlow, GradientBasedOpticalFlow, TranslationDIC

__version__ = '0.11'

class pyIDI:
    def __init__(self, cih_file):
        self.cih_file = cih_file

        self.avaliable_methods = {
            'simplified_optical_flow': SimplifiedOpticalFlow,
            'sof': SimplifiedOpticalFlow,
            'gradient_based_optical_flow': GradientBasedOpticalFlow,
            'gb': GradientBasedOpticalFlow,
        }

        self.mraw, self.info = self.load_video()


    def set_method(self, method, **kwargs):
        """
        Set displacement identification method on video.

        :param method: the method to be used for displacement identification.
        :type method: IDIMethod or str
        """
        if method in self.avaliable_methods.keys():
            self.method = self.avaliable_methods[method](self, **kwargs)
        else:
            self.method = method(self, **kwargs)


    def set_points(self, points=None, method='simplified_optical_flow', **kwargs):
        """
        Set points that will be used to calculate displacements.
        If `points` is None and a `method` has aready been set on this `pyIDI` instance, 
        the `method` object's `get_point` is used to get method-appropriate points.
        """

        if points is None:
            if hasattr(self, 'method'):
                self.method.get_points(self, **kwargs) # get_points sets the attribute video.points
            else:
                raise ValueError("Invalid arguments. Please input points, or set the IDI method first.")
        else:
            self.points = points


    def show_points(self):
        """Show selected points on image.
        """
        if hasattr(self, 'method') and hasattr(self.method, 'show_points'):
            self.method.show_points(self)

        else:
            fig, ax = plt.subplots(figsize=(15, 5))
            ax.imshow(self.mraw[0].astype(float), cmap='gray')
            ax.scatter(self.points[:, 1], self.points[:, 0], marker='.', color='r')
            plt.grid(False)
            plt.show()


    def get_displacements(self, **kwargs):
        """Calculate the displacements based on chosen method.
        """
        if hasattr(self, 'method'):
            self.method.calculate_displacements(self, **kwargs)
            return self.method.displacements
        else:
            raise ValueError('IDI method has not yet been set. Please call `set_method()` first.')


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
        else:
            raise Exception(f'Unknown bit depth: {bit}. Bit depth of the video must be either 8 or 16.\nPlease use correct export options from Photron software')

        filename = '.'.join(self.cih_file.split('.')[:-1])
        mraw = np.memmap(filename+'.mraw', dtype=self.bit_dtype, mode='r', shape=(self.N, self.image_height, self.image_width))
        return mraw, info


    def close_video(self):
        """
        Close the .mraw video memmap.
        """
        if hasattr(self, 'mraw'):
            self.mraw._mmap.close()
            del self.mraw

    
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




    
