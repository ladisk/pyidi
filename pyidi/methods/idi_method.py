import os
import pickle
import numpy as np
import datetime
import json
import glob

from ..video_reader import VideoReader

class IDIMethod:
    """Common functions for all methods.
    """
    
    def __init__(self, video: VideoReader, *args, **kwargs):
        """
        The image displacement identification method constructor.

        For more configuration options, see `method.configure()`
        """
        self.video = video
        self.configure(*args, **kwargs)

    def method_name(self):
        return self.__class__.__name__
    
    def configure(self, *args, **kwargs):
        """
        Configure the displacement identification method here.
        """
        pass

    
    def calculate_displacements(self, video, *args, **kwargs):
        """
        Calculate the displacements of set points here.
        The result should be saved into the `self.displacements` attribute.
        """
        raise NotImplementedError("The 'calculate_displacements' method is not implemented.")
    
    def create_temp_files(self):
        pass
    
    def clear_temp_files(self):
        pass

    def create_settings_dict(self):
        settings = dict()
        return settings
    
    def set_points(self, points):
        points = np.array(points)
        if points.shape[1] != 2:
            raise ValueError("Points must have two columns.")
        
        self.points = points

    def get_displacements(self, autosave=True, **kwargs):
        """
        Calculate the displacements based on chosen method.

        :param autosave: Save the results automatically. Default is True.
        :type autosave: bool
        :param kwargs: Additional keyword arguments that are ultimately passed to the ``configure`` method.
        :type kwargs: dict
        """
        self.calculate_displacements(**kwargs)
        
        # auto-save and clearing temp files
        if hasattr(self, 'process_number') and self.process_number == 0:
            if autosave:
                self.create_analysis_directory()
                self.save(root=self.root_this_analysis)

            self.clear_temp_files()
                
        return self.displacements
    

    def create_analysis_directory(self):
        self.root_analysis = os.path.join(self.video.root, f'{self.video.name}_pyidi_analysis')
        if not os.path.exists(self.root_analysis):
            os.mkdir(self.root_analysis)
        
        analyses = glob.glob(os.path.join(self.root_analysis, 'analysis_*/'))
        if analyses:
            last_an = sorted(analyses)[-1]
            print(last_an, last_an.split('\\')[-2])
            n = int(last_an.split('\\')[-2].split('_')[-1])
        else:
            n = 0
        self.root_this_analysis = os.path.join(self.root_analysis, f'analysis_{n+1:0>3.0f}')
        
        os.mkdir(self.root_this_analysis)

    
    def save(self, root=''):
        with open(os.path.join(root, 'results.pkl'), 'wb') as f:
            pickle.dump(self.displacements, f, protocol=-1)
        with open(os.path.join(root, 'points.pkl'), 'wb') as f:
            pickle.dump(self.points, f, protocol=-1)

        out = {
            'info': {
                'width': self.video.image_width,
                'height': self.video.image_height,
                'N': self.video.N
            },
            'createdate': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'input_file': self.video.input_file,
            'settings': self.create_settings_dict(),
            'method': self.method_name()
        }

        with open(os.path.join(root, 'settings.json'), 'w') as f:
            json.dump(out, f, sort_keys=True, indent=4)

    