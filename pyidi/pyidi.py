import numpy as np
import collections
import matplotlib.pyplot as plt

from .methods import IDIMethod, SimplifiedOpticalFlow, GradientBasedOpticalFlow, LucasKanade
from . import tools

__version__ = '0.17'

available_method_shortcuts = [
    ('sof', SimplifiedOpticalFlow),
    ('lk', LucasKanade),
    ('gb', GradientBasedOpticalFlow)
    ]


class pyIDI:
    """
    The pyIDI base class represents the video to be analysed.
    """
    def __init__(self, cih_file):
        self.cih_file = cih_file

        self.available_methods = dict([ 
            (key, {
                'IDIMethod': method,
                'description': method.__doc__,     
            })
            for key, method in available_method_shortcuts
        ])

        # Fill available methods into `set_method` docstring
        available_methods_doc = '\n' + '\n'.join([
            f"'{key}' ({method_dict['IDIMethod'].__name__}): {method_dict['description']}"
            for key, method_dict in self.available_methods.items()
            ])
        tools.update_docstring(self.set_method, added_doc=available_methods_doc)

        # Load selected video
        self.mraw, self.info = self.load_video()


    def set_method(self, method, **kwargs):
        """
        Set displacement identification method on video.
        To configure the method, use `method.configure()`

        Available methods:
        ---
        [Available method names and descriptions go here.]
        ---

        :param method: the method to be used for displacement identification.
        :type method: IDIMethod or str
        """
        if isinstance(method, str) and method in self.available_methods.keys():
            self.method = self.available_methods[method]['IDIMethod'](self, **kwargs)
        elif callable(method) and hasattr(method, 'calculate_displacements'):
            try:
                self.method = method(self, **kwargs)
            except:
                raise ValueError("The input `method` is not a valid `IDIMethod`.")
        else:
            raise ValueError("method must either be a valid name from `available_methods` or an `IDIMethod`.")
        
        # Update `get_displacements` docstring
        tools.update_docstring(self.get_displacements, self.method.calculate_displacements)


    def set_points(self, points=None, method=None, **kwargs):
        """
        Set points that will be used to calculate displacements.
        If `points` is None and a `method` has aready been set on this `pyIDI` instance, 
        the `method` object's `get_point` is used to get method-appropriate points.
        """
        if points is None:
            if not hasattr(self, 'method'):
                if method is not None:
                    self.set_method(method)
                else:
                    raise ValueError("Invalid arguments. Please input points, or set the IDI method first.")
            self.method.get_points(self, **kwargs) # get_points sets the attribute video.points                
        else:
            self.points = points


    def show_points(self):
        """
        Show selected points on image.
        """
        if hasattr(self, 'method') and hasattr(self.method, 'show_points'):
            self.method.show_points(self)
        else:
            fig, ax = plt.subplots(figsize=(15, 5))
            ax.imshow(self.mraw[0].astype(float), cmap='gray')
            ax.scatter(self.points[:, 1], self.points[:, 0], marker='.', color='r')
            plt.grid(False)
            plt.show()


    def show_field(self, field, scale=1., width=0.5):
        """
        Show displacement field on image.
        
        :param field: Field of displacements (number_of_points, 2)
        :type field: ndarray
        :param scale: scale the field, defaults to 1.
        :param scale: float, optional
        :param width: width of the arrow, defaults to 0.5
        :param width: float, optional
        """
        max_L = np.max(field[:, 0]**2 + field[:, 1]**2)

        fig, ax = plt.subplots(1)
        ax.imshow(self.mraw[0], 'gray')
        for i, ind in enumerate(self.points):
            f0 = field[i, 0]
            f1 = field[i, 1]
            alpha = (f0**2 + f1**2) / max_L
            if alpha < 0.2:
                alpha = 0.2
            plt.arrow(ind[1], ind[0], scale*f1, scale*f0, width=width, color='r', alpha=alpha)


    def get_displacements(self, **kwargs):
        """
        Calculate the displacements based on chosen method.

        Method docstring:
        ---
        Method is not set. Please use the `set_method` method.
        ---
        """
        if hasattr(self, 'method'):
            self.method.calculate_displacements(self, **kwargs)
            return self.method.displacements
        else:
            raise ValueError('IDI method has not yet been set. Please call `set_method()` first.')


    def load_video(self):
        """
        Get video and it's information.
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
        """
        Get info from .cih file in path, return it as dict.

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




    
