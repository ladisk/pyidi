import os
import json
import pickle
import warnings

from . import pyidi

def load_analysis(analysis_path, cih_file=None, load_results=True):
    """Load the previous analysis and create a pyIDI object.

    :param analysis_path: Path to analysis folder (e.g. video_pyidi_analysis/analysis_001/)
    :type analysis_path: str
    :param cih_file: new location of the cih file, if None, the location in settings.txt 
        is used, defaults to None
    :type cih_file: str or None, optional
    :param load_results: if False, the displacements are not loaded,
        only points and settings, defaults to True
    :type load_results: bool, optional
    :return: pyIDI object and settings dict
    :rtype: tuple
    """
    with open(os.path.join(analysis_path, 'settings.txt'), 'r') as f:
        settings = json.load(f)
    
    if cih_file is None:
        video = pyidi.pyIDI(settings['cih_file'])
    else:
        video = pyidi.pyIDI(cih_file)
    
    points = pickle.load(open(os.path.join(analysis_path, 'points.pkl'), 'rb'))
    if load_results:
        results = pickle.load(open(os.path.join(analysis_path, 'results.pkl'), 'rb'))
        video.displacements = results

    video.set_points(points)

    if settings['method'] != 'external_method':
        video.set_method(settings['method'])
        video.method.configure(**settings['settings'])

    else:
        warnings.warn('External method was used for computation. Method is not set.')

    return video, settings['settings']

