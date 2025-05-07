from .video_reader import VideoReader
from .methods import SimplifiedOpticalFlow, LucasKanade, DirectionalLucasKanade, IDIMethod
import warnings

available_method_shortcuts = [
    ('sof', SimplifiedOpticalFlow),
    ('lk', LucasKanade),
    ('lk_1D', DirectionalLucasKanade)
    ]

class pyIDI:
    def __init__(self, input_file, root=None):
        """This class is no longer used in the new version of pyIDI.
        

        .. deprecated:: 1.0
            Use ``VideoReader`` and method-specific classes instead.

        In version 1.0 of pyIDI, some changes were made in the API.

        To convert the old code to the new version, you can use the following example:

        Old version:
        
        >>> from pyidi import pyIDI
        >>> video = pyIDI("path/to/file")
        >>> video.set_method(...)
        >>> video.set_points(...)
        >>> video.method.configure(...)
        >>> video.get_displacements()

        New version:
        
        >>> from pyidi import VideoReader, SimplifiedOpticalFlow, LucasKanade
        >>> video = VideoReader("path/to/file")
        
        >>> idi = SimplifiedOpticalFlow(video)
        >>> idi.set_points(...)
        >>> idi.configure(...)
        >>> idi.get_displacements()
        
        For more information, see the documentation at https://pyidi.readthedocs.io/en/latest/
        """
        warnings.warn("This class is no longer used in the new version of pyIDI. For more information, see the documentation at https://pyidi.readthedocs.io/en/latest/", DeprecationWarning)
        self.video = VideoReader(input_file, root=root)
        self.reader = self.video # for compatibility with old code

    def set_method(self, method):
        """This method is no longer used in the new version of pyIDI.
        
        In version 1.0 of pyIDI, some changes were made in the API.

        To convert the old code to the new version, you can use the following example:

        Old version:

        >>> from pyidi import pyIDI
        >>> video = pyIDI("path/to/file")
        >>> video.set_method(...)
        >>> video.set_points(...)
        >>> video.method.configure(...)
        >>> video.get_displacements()

        New version:

        >>> from pyidi import VideoReader, SimplifiedOpticalFlow, LucasKanade
        >>> video = VideoReader("path/to/file")
        
        >>> idi = SimplifiedOpticalFlow(video)
        >>> idi.set_points(...)
        >>> idi.configure(...)
        >>> idi.get_displacements()
        
        For more information, see the documentation at https://pyidi.readthedocs.io/en/latest/
        """
        warnings.warn("This method is no longer used in the new version of pyIDI. For more information, see the documentation at https://pyidi.readthedocs.io/en/latest/", DeprecationWarning)
        if method == 'sof':
            self.method = SimplifiedOpticalFlow(self.video)
            self.method_name = 'sof'
        elif method == 'lk':
            self.method = LucasKanade(self.video)
            self.method_name = 'lk'
        elif method == 'lk_1D':
            self.method = DirectionalLucasKanade(self.video)
            self.method_name = 'lk_1D'
        elif isinstance(method, IDIMethod):
            self.method = method(self.video)
            self.method_name = method.__name__
        else:
            raise ValueError('Invalid method')
        
    def set_points(self, points):
        """This method is no longer used in the new version of pyIDI.
        
        In version 1.0 of pyIDI, some changes were made in the API.

        To convert the old code to the new version, you can use the following example:

        Old version:

        >>> from pyidi import pyIDI
        >>> video = pyIDI("path/to/file")
        >>> video.set_method(...)
        >>> video.set_points(...)
        >>> video.method.configure(...)
        >>> video.get_displacements()

        New version:

        >>> from pyidi import VideoReader, SimplifiedOpticalFlow, LucasKanade
        >>> video = VideoReader("path/to/file")
        
        >>> idi = SimplifiedOpticalFlow(video)
        >>> idi.set_points(...)
        >>> idi.configure(...)
        >>> idi.get_displacements()
        
        For more information, see the documentation at https://pyidi.readthedocs.io/en/latest/
        """
        warnings.warn("This method is no longer used in the new version of pyIDI. For more information, see the documentation at https://pyidi.readthedocs.io/en/latest/", DeprecationWarning)
        if not hasattr(self, 'method'):
            raise ValueError('Please first set the method')
        self.method.set_points(points)

    def get_displacements(self, *args, **kwargs):
        """This method is no longer used in the new version of pyIDI.
        
        In version 1.0 of pyIDI, some changes were made in the API.

        To convert the old code to the new version, you can use the following example:

        Old version:

        >>> from pyidi import pyIDI
        >>> video = pyIDI("path/to/file")
        >>> video.set_method(...)
        >>> video.set_points(...)
        >>> video.method.configure(...)
        >>> video.get_displacements()

        New version:

        >>> from pyidi import VideoReader, SimplifiedOpticalFlow, LucasKanade
        >>> video = VideoReader("path/to/file")

        >>> idi = SimplifiedOpticalFlow(video)
        >>> idi.set_points(...)
        >>> idi.configure(...)
        >>> idi.get_displacements()
        
        For more information, see the documentation at https://pyidi.readthedocs.io/en/latest/
        """
        warnings.warn("This method is no longer used in the new version of pyIDI. For more information, see the documentation at https://pyidi.readthedocs.io/en/latest/", DeprecationWarning)
        return self.method.get_displacements(*args, **kwargs)