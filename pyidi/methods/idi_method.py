import numpy as np


class IDIMethod:
    """Common functions for all methods.
    """

    def check_kwargs(self, kwargs, options):
        """Check if the given kwargs are valid.
        
        :param kwargs: kwargs
        :type kwargs: dict
        :param options: avaliable options - the kwargs that are valid
        :type options: dict
        """
        for kwarg in kwargs.keys():
            if kwarg not in options.keys():
                raise Exception(f'keyword argument "{kwarg}" is not one of the options for this method')
    
    def change_docstring(self, what_method, options):
        """Add avaliable kwargs to the docstring of `set_method` method.
        """
        docstring = what_method.__doc__.split('---')
        docstring[1] = '- ' + '\n\t- '.join(options) + '\n\t'

        what_method.__func__.__doc__ = '---\n\t'.join(docstring)

