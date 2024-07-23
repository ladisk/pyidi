import numpy as np
import time
import datetime
import os
import shutil
import json
import glob
import warnings

import scipy.signal
from scipy.linalg import lu_factor, lu_solve
from scipy.interpolate import RectBivariateSpline
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from multiprocessing import Pool
import pickle
import numba as nb

from atpbar import atpbar
import mantichora

from psutil import cpu_count
from .. import pyidi
from .. import tools

from .idi_method import IDIMethod
import warnings


class LucasKanade_1D(IDIMethod):
    """
    Translation identification based on the Lucas-Kanade method using least-squares
    iterative optimization with the Zero Normalized Cross Correlation optimization
    criterium.
    """  
    def configure(
        self, roi_size=(9, 9), dij = (1,0), pad=(2,2), max_nfev=20, 
        tol=1e-8, int_order=3, verbose=1, show_pbar=True, 
        processes=1, pbar_type='atpbar', multi_type='mantichora',
        resume_analysis=True, process_number=0, reference_image=0,
        mraw_range='full', use_numba=False
    ):
        """
        Displacement identification based on Directional Lucas-Kanade method,
        using iterative least squares optimization of translatory transformation
        parameters to determine image ROI translations.
        
        :param video: parent object
        :type video: object
        :param roi_size: (h, w) height and width of the region of interest.
            ROI dimensions should be odd numbers. Defaults to (9, 9)
        :type roi_size: tuple, list, optional
        :param dij: Assumed vibration direction. Must be |d|=1. Convention is 'negative down, positive right'.
        :type dij: tuple, list, optional
        :param pad: size of padding around the region of interest in px, defaults to 2
        :type pad: int, optional
        :param max_nfev: maximum number of iterations in least-squares optimization, 
            defaults to 20
        :type max_nfev: int, optional
        :param tol: tolerance for termination of the iterative optimization loop.
            The minimum value of the optimization parameter vector norm.
        :type tol: float, optional
        :param int_order: interpolation spline order
        :type int_order: int, optional
        :param verbose: show text while running, defaults to 1
        :type verbose: int, optional
        :param show_pbar: show progress bar, defaults to True
        :type show_pbar: bool, optional
        :param processes: number of processes to run
        :type processes: int, optional, defaults to 1.
        :param pbar_type: type of the progress bar ('tqdm' or 'atpbar'), defaults to 'atpbar'
        :type pbar_type: str, optional
        :param multi_type: type of multiprocessing used ('multiprocessing' or 'mantichora'), defaults to 'mantichora'
        :type multi_type: str, optional
        :param resume_analysis: if True, the last analysis results are loaded and computation continues from last computed time point.
        :type resum_analysis: bool, optional
        :param process_number: User should not change this (for multiprocessing purposes - to indicate the process number)
        :type process_number: int, optional
        :param reference_image: The reference image for computation. Can be index of a frame, tuple (slice) or numpy.ndarray that
            is taken as a reference.
        :type reference_image: int or tuple or ndarray
        :param mraw_range: Part of the video to process. If "full", a full video is processed. If first element of tuple is not 0,
            a appropriate reference image should be chosen.
        :type mraw_range: tuple or "full"
        :param use_numba: Use numba.njit for computation speedup. Currently not implemented.
        :type use_numba: bool
        """
        if pad is not None:
            self.pad = pad
        if max_nfev is not None:
            self.max_nfev = max_nfev
        if tol is not None:
            self.tol = tol
        if verbose is not None:
            self.verbose = verbose
        if show_pbar is not None:
            self.show_pbar = show_pbar
        if roi_size is not None:
            self.roi_size = roi_size
        if int_order is not None:
            self.int_order = int_order
        if pbar_type is not None:
            self.pbar_type = pbar_type
        if multi_type is not None:
            self.multi_type = multi_type
        if processes is not None:
            self.processes = processes
        if resume_analysis is not None:
            self.resume_analysis = resume_analysis
        if process_number is not None:
            self.process_number = process_number
        if reference_image is not None:
            self.reference_image = reference_image
        if mraw_range is not None:
            self.mraw_range = mraw_range
        if use_numba is not None:
            self.use_numba = use_numba
        if dij is not None:
            self.dij = np.array(dij)
            if np.linalg.norm(self.dij) != 1:
                self.dij = self.dij/np.linalg.norm(self.dij)
                warnings.warn('The direction vector d must have a norm of 1. The input vector was normalized.')
        self._set_mraw_range()

        self.temp_dir = os.path.join(self.video.reader.root, 'temp_file')
        self.settings_filename = os.path.join(self.temp_dir, 'settings.pkl')
        self.analysis_run = 0
        

    def _set_mraw_range(self):
        """Set the range of the video to be processed.
        """
        self.step_time = 1

        if self.mraw_range == 'full':
            self.start_time = 1
            self.stop_time = self.video.reader.N
            
        elif type(self.mraw_range) == tuple:
            if len(self.mraw_range) >= 2:
                if self.mraw_range[0] < self.mraw_range[1] and self.mraw_range[0] > 0:
                    self.start_time = self.mraw_range[0] + self.step_time
                    
                    if self.mraw_range[1] <= self.video.reader.N:
                        self.stop_time = self.mraw_range[1]
                    else:
                        raise ValueError(f'mraw_range can only go to end of video - index {self.video.reader.N}')
                else:
                    raise ValueError(f'Wrong mraw_range definition.')

                if len(self.mraw_range) == 3:
                    self.step_time = self.mraw_range[2]

            else:
                raise Exception('Wrong definition of mraw_range.')
        else:
            raise TypeError(f'mraw_range must be a tuple of start and stop index or "full" ({type(self.mraw_range)}')
            
        self.N_time_points = len(range(self.start_time-self.step_time, self.stop_time, self.step_time))


    def calculate_displacements(self, video, **kwargs):
        """
        Calculate displacements for set points and roi size.

        kwargs are passed to `configure` method. Pre-set arguments (using configure)
        are NOT changed!
        
        """
        # Updating the atributes
        config_kwargs = dict([(var, None) for var in self.configure.__code__.co_varnames])
        config_kwargs.pop('self', None)
        config_kwargs.update((k, kwargs[k]) for k in config_kwargs.keys() & kwargs.keys())
        self.configure(**config_kwargs)

        if self.process_number == 0:
            # Happens only once per analysis
            if self.temp_files_check() and self.resume_analysis:
                if self.verbose:
                    print('-- Resuming last analysis ---')
                    print(' ')
            else:
                self.resume_analysis = False
                if self.verbose:
                    print('--- Starting new analysis ---')
                    print(' ')

        if self.processes != 1:
            if not self.resume_analysis:
                self.create_temp_files(init_multi=True)
            
            self.displacements = multi(video, self.processes)
            # return?

        else:
            self.image_size = (video.reader.image_height, video.reader.image_width)

            if self.resume_analysis:
                self.resume_temp_files()
            else:
                self.displacements = np.zeros((video.points.shape[0], self.N_time_points, 2))
                self.create_temp_files(init_multi=False)

            self.warnings = []

            # Precomputables
            start_time = time.time()

            if self.verbose:
                t = time.time()
                print(f'Interpolating the reference image...')
            self._interpolate_reference(video)
            if self.verbose:
                print(f'...done in {time.time() - t:.2f} s')

            # Time iteration.
            for ii, i in enumerate(self._pbar_range(self.start_time, self.stop_time, self.step_time)):
                ii = ii + 1

                # Iterate over points.
                for p, point in enumerate(video.points):
                    
                    # start optimization with previous optimal parameter values
                    d_init = np.round(self.displacements[p, ii-1, :]).astype(int)
                    d_res = self.displacements[p, ii-1, :] - d_init

                    yslice, xslice = self._padded_slice(point+d_init, self.roi_size, self.image_size, (1,1))
                    G = video.reader.get_frame(i)[yslice, xslice].astype(np.float64)

                    displacements = self.optimize_translations(
                        G=G, 
                        F_spline=self.interpolation_splines[p], 
                        maxiter=self.max_nfev,
                        tol=self.tol,
                        dij = self.dij,
                        d_subpixel_init = -d_res
                        )
                    self.displacements[p, ii, :] = displacements + d_init

                # temp
                self.temp_disp[:, ii, :] = self.displacements[:, ii, :]
                self.update_log(ii)
                    
            del self.temp_disp

            if self.verbose:
                full_time = time.time() - start_time
                if full_time > 60:
                    full_time_m = full_time//60
                    full_time_s = full_time%60
                    print(f'Time to complete: {full_time_m:.0f} min, {full_time_s:.1f} s')
                else:
                    print(f'Time to complete: {full_time:.1f} s')


    def optimize_translations(self, G, F_spline, maxiter, tol, dij, d_subpixel_init=(0, 0)):
        """
        Determine the optimal translation parameters to align the current
        image subset `G` with the interpolated reference image subset `F`.
        
        :param G: the current image subset. (G already is of type float64)
        :type G: array of shape `roi_size`
        :param F_spline: interpolated referencee image subset
        :type F_spline: scipy.interpolate.RectBivariateSpline
        :param maxiter: maximum number of iterations
        :type maxiter: int
        :param tol: convergence criterium
        :type tol: float
        :param d_subpixel_init: initial subpixel displacement guess, 
            relative to the integrer position of the image subset `G`
        :type d_init: array-like of size 2, optional, defaults to (0, 0)
        :return: the obtimal subpixel translation parameters of the current
            image, relative to the position of input subset `G`.
        :rtype: array of size 2
        """
        Gx, Gy  = tools.get_gradient(G) #(Gj, Gi)
        Gd      = Gx*dij[1] + Gy*dij[0]
        Gd2     = np.sum(Gd**2)
        if Gd2 == 0:
            warnings.warn('Division by zero encountered in optimize_translations.')
            Gd2_inv = 0
        else: 
            Gd2_inv = 1/Gd2
        G_clipped = G[1:-1, 1:-1]

        # initialize values
        error = 1.
        displacement = np.array(d_subpixel_init, dtype=np.float64)
        delta = displacement.copy()

        y_f = np.arange(self.roi_size[0], dtype=np.float64)
        x_f = np.arange(self.roi_size[1], dtype=np.float64)

        # optimization loop
        for _ in range(maxiter):
            y_f += delta[0]
            x_f += delta[1]

            F = F_spline(y_f, x_f)
            delta, error = compute_delta_numba(F, G_clipped, Gd, Gd2_inv, dij = dij)

            displacement += delta
            if error < tol:
                return -displacement # roles of F and G are switched

        # max_iter was reached before the convergence criterium
        return -displacement


    def _padded_slice(self, point, roi_size, image_shape, pad=None):
        '''
        Returns a slice that crops an image around a given `point` center, 
        `roi_size` and `pad` size. If the resulting slice would be out of
        bounds of the image to be sliced (given by `image_shape`), the
        slice is snifted to be on the image edge and a warning is issued.
        :param point: The center point coordiante of the desired ROI.
        :type point: array_like of size 2, (y, x)
        :param roi_size: Size of desired cropped image (y, x).
        type roi_size: array_like of size 2, (h, w)
        :param image_shape: Shape of the image to be sliced, (h, w).
        type image_shape: array_like of size 2, (h, w)
        :param pad: Pad border size in pixels. If None, the video.pad
            attribute is read.
        :type pad: int, optional, defaults to None
        :return crop_slice: tuple (yslice, xslice) to use for image slicing.
        '''

        if pad is None:
            pad = self.pad
        y_, x_ = np.array(point).astype(int)
        h, w = np.array(roi_size).astype(int)

        # Bounds checking
        y = np.clip(y_, h//2+pad[0], image_shape[0]-(h//2+pad[0]+1))
        x = np.clip(x_, w//2+pad[1], image_shape[1]-(w//2+pad[1]+1))

        if x != x_ or y != y_:
            warnings.warn('Reached image edge. The displacement optimization ' +
                'algorithm may not converge, or selected points might be too close ' + 
                'to image border. Please check analysis settings.')

        yslice = slice(y-h//2-pad[0], y+h//2+pad[0]+1)
        xslice = slice(x-w//2-pad[1], x+w//2+pad[1]+1)
        return yslice, xslice


    def _pbar_range(self, *args, **kwargs):
        """
        Set progress bar range or normal range.
        """
        if self.show_pbar:
            if self.pbar_type == 'tqdm':
                return tqdm(range(*args, **kwargs), ncols=100, leave=True)
            elif self.pbar_type == 'atpbar':
                try:
                    return atpbar(range(*args, **kwargs), name=f'{self.video.points.shape[0]} points', time_track=True)
                except:
                    return atpbar(range(*args, **kwargs), name=f'{self.video.points.shape[0]} points')
        else:
            return range(*args, **kwargs)


    def _set_reference_image(self, video, reference_image):
        """Set the reference image.
        """
        if type(reference_image) == int:
            ref = video.reader.get_frame(reference_image).astype(float)

        elif type(reference_image) == tuple:
            if len(reference_image) == 2:
                ref = np.zeros((video.reader.image_height, video.reader.image_width), dtype=float)
                for frame in range(reference_image[0], reference_image[1]):
                    ref += video.reader.get_frame(frame)
                ref /= (reference_image[1] - reference_image[0])
  
        elif type(reference_image) == np.ndarray:
            ref = reference_image

        else:
            raise Exception('reference_image must be index of frame, tuple (slice) or ndarray.')
        
        return ref

    def _interpolate_reference(self, video):
        """
        Interpolate the reference image.

        Each ROI is interpolated in advanced to save computation costs.
        Meshgrid for every ROI (without padding) is also determined here and 
        is later called in every time iteration for every point.
        
        :param video: parent object
        :type video: object
        """
        pad = self.pad
        f = self._set_reference_image(video, self.reference_image)
        splines = []
        for point in video.points:
            yslice, xslice = self._padded_slice(point, self.roi_size, self.image_size, pad)

            img_subset = f[yslice, xslice]

            spl = RectBivariateSpline(
               x=np.arange(-pad[1], self.roi_size[0]+pad[1]),
               y=np.arange(-pad[0], self.roi_size[1]+pad[0]),
               z=img_subset,
               kx=self.int_order,
               ky=self.int_order,
               s=0
            )
            splines.append(spl)
        self.interpolation_splines = splines

    @property
    def roi_size(self):
        """
        `roi_size` attribute getter
        """
        return self._roi_size
        
    @roi_size.setter
    def roi_size(self, size):
        """
        ROI size setter. The values in `roi_size` must be odd integers. If not,
        the inputs will be rounded to nearest valid values.
        """
        size = (np.array(size)//2 * 2 + 1).astype(int)
        if np.ndim(size) == 0:
            self._roi_size = np.repeat(size, 2)
        elif np.ndim(size) == 1 and np.size(size) == 2:
            self._roi_size = size
        else:
            raise ValueError(f'Invalid input. ROI size must be scalar or a size 2 array-like.')

    
    def show_points(self, video, figsize=(15, 5), cmap='gray', color='r'):
        """
        Shoe points to be analyzed, together with ROI borders.
        
        :param figsize: matplotlib figure size, defaults to (15, 5)
        :param cmap: matplotlib colormap, defaults to 'gray'
        :param color: marker and border color, defaults to 'r'
        """
        roi_size = self.roi_size

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(video.reader.get_frame(0).astype(float), cmap=cmap)
        ax.scatter(video.points[:, 1],
                   video.points[:, 0], marker='.', color=color)

        for point in video.points:
            roi_border = patches.Rectangle((point - self.roi_size//2 - 0.5)[::-1], self.roi_size[1], self.roi_size[0],
                                            linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(roi_border)

        plt.grid(False)
        plt.show()


    def create_temp_files(self, init_multi=False):
        """Temporary files to track the solving process.

        This is done in case some error occures. In this eventuality the calculation
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
                pickle.dump(self.video.points, f)

        if not init_multi:
            token = f'{self.process_number:0>3.0f}'

            self.process_log = os.path.join(temp_dir, 'process_log_' + token + '.txt')
            self.points_filename = os.path.join(temp_dir, 'points.pkl')
            self.disp_filename = os.path.join(temp_dir, 'disp_' + token + '.pkl')

            with open(self.process_log, 'w', encoding='utf-8') as f:
                f.writelines([
                    f'cih_file: {self.video.cih_file}\n',
                    f'token: {token}\n',
                    f'points_filename: {self.points_filename}\n',
                    f'disp_filename: {self.disp_filename}\n',
                    f'disp_shape: {(self.video.points.shape[0], self.N_time_points, 2)}\n',
                    f'analysis_run <{self.analysis_run}>:'
                ])

            self.temp_disp = np.memmap(self.disp_filename, dtype=np.float64, mode='w+', shape=(self.video.points.shape[0], self.N_time_points, 2))
            

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
            settings_old = pickle.load(open(self.settings_filename, 'rb'))
            json_old = json.dumps(settings_old, sort_keys=True, indent=2)
            
            settings_new = self._make_comparison_dict()
            json_new = json.dumps(settings_new, sort_keys=True, indent=2)

            # if settings are different - new analysis
            if json_new != json_old:
                return False
            
            # if points file exists and points are the same
            if os.path.exists(os.path.join(self.temp_dir, 'points.pkl')):
                points = pickle.load(open(os.path.join(self.temp_dir, 'points.pkl'), 'rb'))
                if np.array_equal(points, self.video.points):
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
        INCLUDE_KEYS = [
            '_roi_size',
            'dij',
            'pad',
            'max_nfev',
            'tol',
            'int_order',
            'show_pbar',
            'processes',
            'pbar_type',
            'multi_type',
            'reference_image',
            'mraw_range',
        ]

        settings = dict()
        data = self.__dict__
        for k, v in data.items():
            if k in INCLUDE_KEYS:
                if k == '_roi_size':
                    k = 'roi_size'
                if type(v) in [int, float, str]:
                    settings[k] = v
                elif type(v) in [list, tuple]:
                    if len(v) < 10:
                        settings[k] = v
                elif type(v) == np.ndarray:
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
            # 'configure': dict([(var, None) for var in self.configure.__code__.co_varnames]),
            'configure': self.create_settings_dict(),
            'info': {
                'width': self.video.reader.image_width,
                'height': self.video.reader.image_height,
                'N': self.video.reader.N
            }
        }
        return settings


    @staticmethod
    def get_points():
        raise Exception('Choose a method from `tools` module.')


def multi(video, processes):
    """
    Splitting the points to multiple processes and creating a
    pool of workers.
    
    :param video: the video object with defined attributes
    :type video: object
    :param processes: number of processes. If negative, the number
        of processes is set to `psutil.cpu_count + processes`.
    :type processes: int
    :return: displacements
    :rtype: ndarray
    """
    if processes < 0:
        processes = cpu_count() + processes
    elif processes == 0:
        raise ValueError('Number of processes must not be zero.')

    points = video.points
    points_split = tools.split_points(points, processes=processes)
    
    idi_kwargs = {
        'cih_file': video.cih_file,
    }
    
    method_kwargs = {
        'roi_size': video.method.roi_size, 
        'pad': video.method.pad, 
        'dij': video.method.dij,
        'max_nfev': video.method.max_nfev, 
        'tol': video.method.tol, 
        'verbose': video.method.verbose, 
        'show_pbar': video.method.show_pbar,
        'int_order': video.method.int_order,
        'pbar_type': video.method.pbar_type,
        'resume_analysis': video.method.resume_analysis,
        'reference_image': video.method.reference_image,
        'mraw_range': video.method.mraw_range,
    }
    if video.method.pbar_type == 'atpbar':
        print(f'Computation start: {datetime.datetime.now()}')
    t_start = time.time()

    if video.method.multi_type == 'multiprocessing':
        if method_kwargs['pbar_type'] == 'atpbar':
            method_kwargs['pbar_type'] = 'tqdm'
            warnings.warn('"atpbar" pbar_type was used with "multiprocessing". This is not supported. Changed pbar_type to "tqdm"')

        pool = Pool(processes=processes)
        results = [pool.apply_async(worker, args=(p, idi_kwargs, method_kwargs, i)) for i, p in enumerate(points_split)]
        pool.close()
        pool.join()

        out = []
        for r in results:
            out.append(r.get())

        out1 = sorted(out, key=lambda x: x[1])
        out1 = np.concatenate([d[0] for d in out1])
    
    elif video.method.multi_type == 'mantichora':
        with mantichora.mantichora(nworkers=processes) as mcore:
            for i, p in enumerate(points_split):
                mcore.run(worker, p, idi_kwargs, method_kwargs, i)
            returns = mcore.returns()
        
        out = []
        for r in returns:
            out.append(r)
        
        out1 = sorted(out, key=lambda x: x[1])
        out1 = np.concatenate([d[0] for d in out1])


    t = time.time() - t_start
    minutes = t//60
    seconds = t%60
    hours = minutes//60
    minutes = minutes%60
    print(f'Computation duration: {hours:0>2.0f}:{minutes:0>2.0f}:{seconds:.2f}')
    
    return out1


def worker(points, idi_kwargs, method_kwargs, i):
    """
    A function that is called when for each job in multiprocessing.
    """
    method_kwargs['process_number'] = i+1
    _video = pyidi.pyIDI(**idi_kwargs)
    _video.set_method(LucasKanade_1D)
    _video.method.configure(**method_kwargs)
    _video.set_points(points)
    
    return _video.get_displacements(verbose=0), i

# @nb.njit
def compute_delta_numba(F, G, Gd, Gd2_inv, dij):
    F_G = G - F
    error = np.sum(Gd*F_G)*Gd2_inv
    delta = dij*error
    return delta, error
