import os
import json
import pickle
import warnings

from .methods import LucasKanade, SimplifiedOpticalFlow, DirectionalLucasKanade, IDIMethod
from .video_reader import VideoReader

method_mappings = {
    "LucasKanade": LucasKanade,
    "SimplifiedOpticalFlow": SimplifiedOpticalFlow,
    "DirectionalLucasKanade": DirectionalLucasKanade,
}

def load_analysis(analysis_path, input_file=None, load_results=True):
    """Load the previous analysis and create a pyIDI object.

    :param analysis_path: Path to analysis folder (e.g. video_pyidi_analysis/analysis_001/)
    :type analysis_path: str
    :param input_file: new location of the cih file, if None, the location in settings.txt 
        is used, defaults to None
    :type input_file: str or None, optional
    :param load_results: if False, the displacements are not loaded,
        only points and settings, defaults to True
    :type load_results: bool, optional
    :return: pyIDI object and settings dict
    :rtype: tuple
    """
    with open(os.path.join(analysis_path, 'settings.json'), 'r') as f:
        settings = json.load(f)
    
    if input_file is None:
        video = VideoReader(settings['input_file'])
    else:
        video = VideoReader(input_file)

    method_name = settings['method']
    if method_name not in method_mappings:
        raise ValueError(f"Method {method_name} not one of {list(method_mappings.keys())}")
    
    idi: IDIMethod = method_mappings[method_name](video)
    
    with open(os.path.join(analysis_path, 'points.pkl'), 'rb') as f:
        points = pickle.load(f)

    if load_results:
        with open(os.path.join(analysis_path, 'results.pkl'), 'rb') as f:
            results = pickle.load(f)
            
        idi.displacements = results

    idi.set_points(points)

    return video, idi, settings['settings']

