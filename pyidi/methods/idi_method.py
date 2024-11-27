import os
import pickle
import numpy as np
import datetime
import json
import glob
import shutil
import inspect
import matplotlib.pyplot as plt

from ..selection import SubsetSelection
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
        self.process_number = 0
        self.configure(*args, **kwargs)

        # Set the temporary directory
        self.temp_dir = os.path.join(self.video.root, 'temp_file')
        self.settings_filename = os.path.join(self.temp_dir, 'settings.pkl')
        self.analysis_run = 0

    def method_name(self):
        return self.__class__.__name__
    
    def configure(self):
        """
        Configure the displacement identification method here.

        IMPORTANT:
        ----------
        All of the settings should be passed through this method.
        Inside the method, each setting should be saved as an attribute of the class, 
        **keeping the same name as the argument**. This is important as the settings are then saved to
        the results file and can be used to reproduce the results.

        See the ``LucasKanade`` example for example usage.
        """
        pass

    def configure_multiprocessing(self, process_number, progress, task_id):
        """
        Configure the multiprocessing settings here.

        :param process_number: The number of the process.
        :type process_number: int
        :param progress: The progress object.
        :type progress: multiprocessing.Value
        :param task_id: The task ID.
        :type task_id: multiprocessing.Value
        """
        self.process_number = process_number
        self.progress = progress
        self.task_id = task_id

    def calculate_displacements(self):
        """
        Calculate the displacements of set points here.
        The result should be saved into the `self.displacements` attribute.
        """
        raise NotImplementedError("The 'calculate_displacements' method is not implemented.")
    
    def create_temp_files(self, init_multi=False):
        """Temporary files to track the solving process.

        This is done in case some error occurs. In this eventuality the calculation
        can be resumed from the last computed time point.
        
        :param init_multi: when initialization multiprocessing, defaults to False
        :type init_multi: bool, optional
        """
        temp_dir = self.temp_dir
        
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)
        else:
            if self.process_number == 0:
                shutil.rmtree(temp_dir)
                os.mkdir(temp_dir)
        
        if self.process_number == 0:
            # Write all the settings of the analysis
            settings = self._make_comparison_dict()
            with open(self.settings_filename, 'wb') as f:
                pickle.dump(settings, f)
            
            self.points_filename = os.path.join(temp_dir, 'points.pkl')
            with open(self.points_filename, 'wb') as f:
                pickle.dump(self.points, f)

        if not init_multi:
            token = f'{self.process_number:0>3.0f}'

            self.process_log = os.path.join(temp_dir, 'process_log_' + token + '.txt')
            self.points_filename = os.path.join(temp_dir, 'points.pkl')
            self.disp_filename = os.path.join(temp_dir, 'disp_' + token + '.pkl')

            with open(self.process_log, 'w', encoding='utf-8') as f:
                f.writelines([
                    f'input_file: {self.video.input_file}\n',
                    f'token: {token}\n',
                    f'points_filename: {self.points_filename}\n',
                    f'disp_filename: {self.disp_filename}\n',
                    f'disp_shape: {(self.points.shape[0], self.N_time_points, 2)}\n',
                    f'analysis_run <{self.analysis_run}>:'
                ])

            self.temp_disp = np.memmap(self.disp_filename, dtype=np.float64, mode='w+', shape=(self.points.shape[0], self.N_time_points, 2))
    
    def clear_temp_files(self):
        """Clearing the temporary files.
        """
        shutil.rmtree(self.temp_dir)

    def update_log(self, last_time):
        """Updating the log file. 

        A new last time is written in the log file in order to
        track the solution process.
        
        :param last_time: Last computed time point (index)
        :type last_time: int
        """
        with open(self.process_log, 'r', encoding='utf-8') as f:
            log = f.readlines()
        
        log_entry = f'analysis_run <{self.analysis_run}>: finished: {datetime.datetime.now()}\tlast time point: {last_time}'
        if f'<{self.analysis_run}>' in log[-1]:
            log[-1] = log_entry
        else:
            log.append('\n' + log_entry)

        with open(self.process_log, 'w', encoding='utf-8') as f:
            f.writelines(log)

    def resume_temp_files(self):
        """Reload the settings written in the temporary files.

        When resuming the computation of displacement, the settings are
        loaded from the previously created temporary files.
        """
        temp_dir = self.temp_dir
        token = f'{self.process_number:0>3.0f}'

        self.process_log = os.path.join(temp_dir, 'process_log_' + token + '.txt')
        self.disp_filename = os.path.join(temp_dir, 'disp_' + token + '.pkl')

        with open(self.process_log, 'r', encoding='utf-8') as f:
            log = f.readlines()

        shape = tuple([int(_) for _ in log[4].replace(' ', '').split(':')[1].replace('(', '').replace(')', '').split(',')])
 
        self.temp_disp = np.memmap(self.disp_filename, dtype=np.float64, mode='r+', shape=shape)
        self.displacements = np.array(self.temp_disp).copy()

        self.start_time = int(log[-1].replace(' ', '').rstrip().split('\t')[1].split(':')[1]) + 1
        self.analysis_run = int(log[-1].split('<')[1].split('>')[0]) + 1

    def temp_files_check(self):
        """Checking the settings of computation.

        The computation can only be resumed if all the settings and data
        are the same as with the original analysis.
        This function checks that (writing all the setting to dict and
        comparing the json dump of the dicts).

        If the settings are the same but the points are not, a new analysis is
        also started. To set the same points, check the `temp_pyidi` folder.
        
        :return: Whether to resume analysis or not
        :rtype: bool
        """
        # if settings file exists
        if os.path.exists(self.settings_filename):
            with open(self.settings_filename, 'rb') as f:
                settings_old = pickle.load(f)
            json_old = json.dumps(settings_old, sort_keys=True, indent=2)
            
            settings_new = self._make_comparison_dict()
            json_new = json.dumps(settings_new, sort_keys=True, indent=2)

            # if settings are different - new analysis
            if json_new != json_old:
                return False
            
            # if points file exists and points are the same
            if os.path.exists(os.path.join(self.temp_dir, 'points.pkl')):
                with open(os.path.join(self.temp_dir, 'points.pkl'), 'rb') as f:
                    points = pickle.load(f)
                if np.array_equal(points, self.points):
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def create_settings_dict(self):
        """Make a dictionary of the chosen settings.
        """
        INCLUDE_KEYS = self.configuration_keys

        settings = dict()
        data = self.__dict__
        for k, v in data.items():
            if k in INCLUDE_KEYS:
                if type(v) in [int, float, str]:
                    settings[k] = v
                elif type(v) in [list, tuple]:
                    if len(v) < 10:
                        settings[k] = v
                elif type(v) is np.ndarray:
                    if v.size < 10:
                        settings[k] = v.tolist()
        return settings
    
    def _make_comparison_dict(self):
        """Make a dictionary for comparing the original settings with the
        current settings.

        Used for finding out if the analysis should be resumed or not.
        
        :return: Settings
        :rtype: dict
        """
        settings = {
            'configure': self.create_settings_dict(),
            'info': {
                'width': self.video.image_width,
                'height': self.video.image_height,
                'N': self.video.N
            }
        }
        return settings
    
    def set_points(self, points):
        if isinstance(points, list):
            points = np.array(points)
        elif isinstance(points, SubsetSelection):
            points = np.array(points.points)

        points = np.array(points)
        if points.shape[1] != 2:
            raise ValueError("Points must have two columns.")
        
        self.points = points

    def show_points(self, figsize=(15, 5), cmap='gray', color='r'):
        """
        Shoe points to be analyzed, together with ROI borders.
        
        :param figsize: matplotlib figure size, defaults to (15, 5)
        :param cmap: matplotlib colormap, defaults to 'gray'
        :param color: marker and border color, defaults to 'r'
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.video.get_frame(0).astype(float), cmap=cmap)
        ax.scatter(self.points[:, 1],
                   self.points[:, 0], marker='.', color=color)

        plt.grid(False)
        plt.show()

    def get_displacements(self, autosave=True, **kwargs):
        """
        Calculate the displacements based on chosen method.

        :param autosave: Save the results automatically. Default is True.
        :type autosave: bool
        :param kwargs: Additional keyword arguments that are ultimately passed to the ``configure`` method.
        :type kwargs: dict
        """
        # Updating the attributes with the new configuration
        config_kwargs = dict([(var, None) for var in self.configure.__code__.co_varnames])
        config_kwargs.pop('self', None)
        config_kwargs.update((k, kwargs[k]) for k in config_kwargs.keys() & kwargs.keys())
        self.configure(**config_kwargs)

        # Get all the keys that the configure method uses (to be able to save the unique settings)
        self.configuration_keys = self.extract_configuration_arguments()

        # Compute the displacements
        self.calculate_displacements()
        
        # auto-save
        if autosave:
            self.create_analysis_directory()
            self.save(root=self.root_this_analysis)

        return self.displacements
    
    def extract_configuration_arguments(self):
        """Extract the configuration arguments from the configure method."""
        signature = inspect.signature(self.configure)

        return list(signature.parameters.keys())

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

    