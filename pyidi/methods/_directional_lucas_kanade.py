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
import pickle
import numba as nb

from psutil import cpu_count
from .. import tools

from .idi_method import IDIMethod
from ..video_reader import VideoReader
from ..progress_bar import progress_bar, rich_progress_bar_setup
import warnings


class DirectionalLucasKanade(IDIMethod):
    """
    Unidirectional translation identification as introduced in:
    Masmeijer T., Habtour E., Zaletelj, K., & Slavič, J., (2024). Directional DIC method with automatic feature selection. MSSP.
    "https://doi.org/10.1016/j.ymssp.2024.112080".
    The implementation is based on the Lucas-Kanade method using least-squares iterative optimization with the Zero Normalized
    Cross Correlation optimization criterium.
    """  
    def configure(
        self, roi_size=(9, 9), dij = (1,0), pad=(2,2), max_nfev=20, 
        tol=1e-8, int_order=3, verbose=1, show_pbar=True, 
        processes=1, resume_analysis=True, reference_image=0,
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
        :param dij: Assumed vibration direction. If \\|d\\| != 1, the vector is normalized. 
            Convention is 'negative down, positive right'.
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
        :param resume_analysis: if True, the last analysis results are loaded and computation continues from last computed time point.
        :type resume_analysis: bool, optional
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
            self.roi_size = np.array(roi_size, dtype=int)
        if int_order is not None:
            self.int_order = int_order
        if processes is not None:
            self.processes = processes
        if resume_analysis is not None:
            self.resume_analysis = resume_analysis
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
                # warnings.warn('The direction vector d must have a norm of 1. The input vector was normalized.')
        self._set_mraw_range()

        self.temp_dir = os.path.join(self.video.root, 'temp_file')
        self.settings_filename = os.path.join(self.temp_dir, 'settings.pkl')
        self.analysis_run = 0
        

    def _set_mraw_range(self):
        """Set the range of the video to be processed.
        """
        self.step_time = 1

        if self.mraw_range == 'full':
            self.start_time = 1
            self.stop_time = self.video.N
            
        elif type(self.mraw_range) == tuple:
            if len(self.mraw_range) >= 2:
                if self.mraw_range[0] < self.mraw_range[1] and self.mraw_range[0] >= 0:
                    self.start_time = self.mraw_range[0] + self.step_time
                    
                    if self.mraw_range[1] <= self.video.N:
                        self.stop_time = self.mraw_range[1]
                    else:
                        raise ValueError(f'mraw_range can only go to end of video - index {self.video.N}')
                else:
                    raise ValueError(f'Wrong mraw_range definition.')

                if len(self.mraw_range) == 3:
                    self.step_time = self.mraw_range[2]

            else:
                raise Exception('Wrong definition of mraw_range.')
        else:
            raise TypeError(f'mraw_range must be a tuple of start and stop index or "full" ({type(self.mraw_range)}')
            
        self.N_time_points = len(range(self.start_time-self.step_time, self.stop_time, self.step_time))


    def calculate_displacements(self, **kwargs):
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
            
            self.displacements = multi(self.video, self, self.processes, self.configuration_keys)
            
            # Clear the temporary files (only once per analysis)
            self.clear_temp_files()
            return

        self.image_size = (self.video.image_height, self.video.image_width)

        if self.resume_analysis:
            self.resume_temp_files()
        else:
            self.displacements = np.zeros((self.points.shape[0], self.N_time_points, 2))
            self.create_temp_files(init_multi=False)

        self.warnings = []

        # Precomputables
        start_time = time.time()

        if self.verbose:
            t = time.time()
            print(f'Interpolating the reference image...')
        self._interpolate_reference(self.video)
        if self.verbose:
            print(f'...done in {time.time() - t:.2f} s')

        # Time iteration.
        len_of_task = len(range(self.start_time, self.stop_time, self.step_time))
        for ii, i in enumerate(progress_bar(self.start_time, self.stop_time, self.step_time, show_pbar=self.show_pbar)):
            ii = ii + 1

            # Iterate over points.
            for p, point in enumerate(self.points):
                
                # start optimization with previous optimal parameter values
                d_init = np.round(self.displacements[p, ii-1, :]).astype(int)
                d_res = self.displacements[p, ii-1, :] - d_init

                yslice, xslice = self._padded_slice(point+d_init, self.roi_size, self.image_size, (1,1))
                G = self.video.get_frame(i)[yslice, xslice].astype(np.float64)

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

            # Update progress bar if multiple processes
            if hasattr(self, "progress") and hasattr(self, "task_id"):
                self.progress[self.task_id] = {"progress": ii + 1, "total": len_of_task}
                
        del self.temp_disp

        if self.verbose:
            full_time = time.time() - start_time
            if full_time > 60:
                full_time_m = full_time//60
                full_time_s = full_time%60
                print(f'Time to complete: {full_time_m:.0f} min, {full_time_s:.1f} s')
            else:
                print(f'Time to complete: {full_time:.1f} s')

        if self.process_number == 0:
            self.clear_temp_files()


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
        """
        Returns a slice that crops an image around a given `point` center, 
        `roi_size` and `pad` size. If the resulting slice would be out of
        bounds of the image to be sliced (given by `image_shape`), the
        slice is snifted to be on the image edge and a warning is issued.
        :param point: The center point coordiante of the desired ROI.
        :type point: array_like of size 2, (y, x)
        :param roi_size: Size of desired cropped image (y, x).
        :type roi_size: array_like of size 2, (h, w)
        :param image_shape: Shape of the image to be sliced, (h, w).
        :type image_shape: array_like of size 2, (h, w)
        :param pad: Pad border size in pixels. If None, the video.pad attribute is read.
        :type pad: int, optional, defaults to None
        :return crop_slice: tuple (yslice, xslice) to use for image slicing.
        """

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

    def _set_reference_image(self, video, reference_image):
        """Set the reference image.
        """
        if type(reference_image) == int:
            ref = video.get_frame(reference_image).astype(float)

        elif type(reference_image) == tuple:
            if len(reference_image) == 2:
                ref = np.zeros((video.image_height, video.image_width), dtype=float)
                for frame in range(reference_image[0], reference_image[1]):
                    ref += video.get_frame(frame)
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
        for point in self.points:
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


    def show_points(self, figsize=(15, 5), cmap='gray', color='r'):
        """
        Shoe points to be analyzed, together with ROI borders.
        
        :param figsize: matplotlib figure size, defaults to (15, 5)
        :param cmap: matplotlib colormap, defaults to 'gray'
        :param color: marker and border color, defaults to 'r'
        """
        roi_size = self.roi_size

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.video.get_frame(0).astype(float), cmap=cmap)
        ax.scatter(self.points[:, 1],
                   self.points[:, 0], marker='.', color=color)

        for point in self.points:
            roi_border = patches.Rectangle((point - self.roi_size//2 - 0.5)[::-1], self.roi_size[1], self.roi_size[0],
                                            linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(roi_border)

        plt.grid(False)
        plt.show()


def multi(video: VideoReader, idi_method: DirectionalLucasKanade, processes, configuration_keys: list):
    """
    Splitting the points to multiple processes and creating a
    pool of workers.
    
    :param video: VideoReader object
    :type video: VideoReader
    :param idi_method: IDIMethod object
    :type idi_method: IDIMethod
    :param processes: number of processes to run
    :type processes: int
    :param configuration_keys: list of configuration keys
    :type configuration_keys: list
    """
    from concurrent.futures import ProcessPoolExecutor
    from rich import progress
    import multiprocessing

    if processes < 0:
        processes = cpu_count() + processes
    elif processes == 0:
        raise ValueError('Number of processes must not be zero.')

    points = idi_method.points
    points_split = tools.split_points(points, processes=processes)

    idi_kwargs = {
        'input_file': video.input_file,
        'root': video.root,
    }

    if video.file_format == 'np.ndarray':
        idi_kwargs['input_file'] = video.mraw # if the input is np.ndarray, the input_file is the actual data
    

    # Set the parameters that are passed to the configure method
    exclude_keys = ["processes"]
    method_kwargs = dict([(k, idi_method.__dict__.get(k, None)) for k in configuration_keys if k not in exclude_keys])

    print(f'Computation start: {datetime.datetime.now()}')

    t_start = time.time()

    with rich_progress_bar_setup() as progress:
        futures = []
        with multiprocessing.Manager() as manager:
            # this is the key - we share some state between our 
            # main process and our worker functions
            _progress = manager.dict()

            with ProcessPoolExecutor(max_workers=processes) as executor:
                for n in range(0, len(points_split)):  # iterate over the jobs we need to run
                    # set visible false so we don't have a lot of bars all at once:
                    task_id = progress.add_task(f"task {n} ({len(points_split[n])} points)")
                    futures.append(executor.submit(worker, points_split[n], idi_kwargs, method_kwargs, n, _progress, task_id))

                # monitor the progress:
                while sum([future.done() for future in futures]) < len(futures):
                    for task_id, update_data in _progress.items():
                        latest = update_data["progress"]
                        total = update_data["total"]
                        # update the progress bar for this task:
                        progress.update(task_id, completed=latest, total=total+1)

                out = []
                for future in futures:
                    out.append(future.result())

                out1 = sorted(out, key=lambda x: x[1])
                out1 = np.concatenate([d[0] for d in out1])

    t = time.time() - t_start
    minutes = t//60
    seconds = t%60
    hours = minutes//60
    minutes = minutes%60
    print(f'Computation duration: {hours:0>2.0f}:{minutes:0>2.0f}:{seconds:.2f}')
    
    return out1


def worker(points, idi_kwargs, method_kwargs, i, progress, task_id):
    """
    A function that is called when for each job in multiprocessing.
    """
    method_kwargs['show_pbar'] = False # use the rich progress bar insted of tqdm
    
    video = VideoReader(**idi_kwargs)
    idi = DirectionalLucasKanade(video)
    idi.configure(**method_kwargs)
    idi.configure_multiprocessing(i+1, progress, task_id) # configure the multiprocessing settings
    idi.set_points(points)
    return idi.get_displacements(autosave=False), i


# @nb.njit
def compute_delta_numba(F, G, Gd, Gd2_inv, dij):
    F_G = G - F
    error = np.sum(Gd*F_G)*Gd2_inv
    delta = dij*error
    return delta, error
