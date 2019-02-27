import numpy as np


class IDIMethod:
    """Common functions for all methods.
    """
    
    def change_docstring(self, what_method, options):
        """Add avaliable kwargs to the docstring of `set_method` method.
        """
        docstring = what_method.__doc__.split('---')
        docstring[1] = '- ' + '\n\t- '.join(options) + '\n\t'

        what_method.__func__.__doc__ = '---\n\t'.join(docstring)

