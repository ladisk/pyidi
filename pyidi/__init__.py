__version__ = '1.0.1'
# from .pyidi import *
from .pyidi_legacy import pyIDI
from . import tools
from . import postprocessing
from .selection import SubsetSelection
from .load_analysis import load_analysis
from .video_reader import VideoReader
from .methods import *
from .gui import GUI