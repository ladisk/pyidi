import numpy as np


class IDIMethod:
    """Common functions for all methods.
    """
    
    def __init__(self, video, *args, **kwargs):
        """
        The image displacement identification method constructor.

        For more configuration options, see `method.configure()`
        """
        self.video = video
        self.configure(*args, **kwargs)
    
    
    def configure(self, *args, **kwargs):
        """
        Configure the displacement identification method here.
        """
        pass

