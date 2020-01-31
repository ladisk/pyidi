import numpy as np
import time

import scipy.signal
from scipy.interpolate import interp2d
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from multiprocessing import Pool
from psutil import cpu_count
from .. import pyidi
from .. import tools

from .idi_method import IDIMethod


class LucasKanadeSc(IDIMethod):
    """
    Translation identification based on the Lucas-Kanade method using least-squares
    iterative optimization from scipy.optimize.least_squares.
    """  
    def configure(
        self, roi_size=9, pad=2, max_nfev=20, tol=1e-8, verbose=1, show_pbar=True
    ):
        """
        Displacement identification based on Lucas-Kanade method.

        Using iterative approach to determine displacements.
        Least-squares method from scipy.optimize is used.
        
        :param video: parent object
        :type video: object
        :param roi_size: size of the region of interest, defaults to 3
        :type roi_size: int, tuple, list, optional
        :param pad: size of padding around the region of interest, defaults to 2
        :type pad: int, optional
        :param max_nfev: maximum number of iterations in least-squares optimization, defaults to 20
        :type max_nfev: int, optional
        :param tol: tolerance for termination (in least_squares 'ftol', 'xtol', 'gtol' are set equal to 'tol')
        :type tol: float, optional
        :param verbose: show text while running, defaults to 1
        :type verbose: int, optional
        :param show_pbar: show progress bar, defaults to True
        :type show_pbar: bool, optional
        """
        self.pad = pad
        self.max_nfev = max_nfev
        self.tol = tol
        self.verbose = verbose
        self.show_pbar = show_pbar
        self.roi_size = roi_size

        self._set_roi_size(self.roi_size)


    def calculate_displacements(self, video, processes=1, roi_size=None, max_nfev=None, tol=None, **kwargs):
        """
        Calculate displacements for set points and roi size.
        
        :param video: parent object
        :type video: object
        :param processes: numebr of processes to run with `multiprocessing`. Defaults to 1.
        :type processes: int, optional. 
        :param roi_size: size of region of interest (if None, predetermined is used), defaults to None
        :type roi_size: None, int, tuple, list, optional
        :param max_nfev: Maximum number of iterations in least_squares (if None, predetermined is used)
        :type max_nfev: None, int, optional
        :param tol: tolerance for termination (in least_squares 'ftol', 'xtol', 'gtol' are set equal to 'tol')
                    (if None, predetermined is used)
        :type tol: float, optional
        """
        if processes != 1:
            self.displacements = multi(video, processes)
        else:

            if 'verbose' in kwargs.keys():
                self.verbose = kwargs['verbose']

            if roi_size is not None:
                self._set_roi_size(roi_size)

            if max_nfev is not None:
                self.max_nfev = max_nfev
            
            if tol is not None:
                self.tol = tol


            def opt(d, p, G):
                """
                Optimization function.
                """
                F_current = self.F_int[p](self.extended_points_0[p, self.pad:-self.pad] - d[0], self.extended_points_1[p, self.pad:-self.pad] - d[1])
                return (F_current - G).flatten()


            self.displacements = np.zeros((video.points.shape[0], video.N, 2))

            start_time = time.time()

            extend_0 = np.arange(0, self.roi_size[0] + 2*self.pad) - self.roi_size[0]//2 - self.pad
            extend_1 = np.arange(0, self.roi_size[1] + 2*self.pad) - self.roi_size[1]//2 - self.pad

            self.extended_points_0 = np.vstack(video.points[:, 0]) + extend_0
            self.extended_points_1 = np.vstack(video.points[:, 1]) + extend_1

            if self.verbose:
                t = time.time()
                print(f'Interpolating the reference image...')
            self._interpolation(video)

            if self.verbose:
                print(f'...done in {time.time() - t:.2f} s')

            # Time iteration.
            for i in self._pbar(1, len(video.mraw)):
                # Iterate over points.
                for p in range(video.points.shape[0]):
                    G = video.mraw[i, self.mgrid_0[p], self.mgrid_1[p]]
                    
                    delta = self.displacements[p, i-1] # Initial value

                    # Optimization
                    sol = scipy.optimize.least_squares(
                        lambda x: opt(x, p, G), x0=delta, 
                        max_nfev=self.max_nfev, xtol=self.tol, ftol=self.tol, gtol=self.tol
                    ) 
                    self.displacements[p, i] = sol.x
            
            if self.verbose:
                full_time = time.time() - start_time
                if full_time > 60:
                    full_time_m = full_time//60
                    full_time_s = full_time%60
                    print(f'Time to complete: {full_time_m:.0f} min, {full_time_s:.1f} s')
                else:
                    print(f'Time to complete: {full_time:.1f} s')
    

    def _pbar(self, x, y):
        """
        Set progress bar range or normal range.
        
        :param x: start
        :param y: stop
        :return: tqdm range or python range
        """
        if self.show_pbar:
            return tqdm(range(x, y), ncols=100, leave=True)
        else:
            return range(x, y)


    def _interpolation(self, video):
        """
        Interpolate the reference image.

        Each ROI is interpolated in advanced to save computation costs.
        Meshgrid for every ROI (without padding) is also determined here and 
        is later called in every time iteration for every point.
        
        :param video: parent object
        :type video: object
        """
        self.F_int = []
        self.mgrid_0 = []
        self.mgrid_1 = []
        for p in range(video.points.shape[0]):
            _m_0, _m_1 = np.meshgrid(self.extended_points_0[p], self.extended_points_1[p])
            _F_int = interp2d(self.extended_points_0[p], self.extended_points_1[p], video.mraw[0, _m_0, _m_1], kind='cubic')
            self.F_int.append(_F_int)

            m_0, m_1 = np.meshgrid(self.extended_points_0[p, self.pad:-self.pad], self.extended_points_1[p, self.pad:-self.pad])
            self.mgrid_0.append(m_0)
            self.mgrid_1.append(m_1)


    def _set_roi_size(self, roi_size):
        """
        Set ROI size for displacement identification.

        :param roi_size: size of the region of interest
        :type roi_size: int, list, tuple
        """
        if isinstance(roi_size, int):
            self.roi_size = np.array([roi_size, roi_size], dtype=int)
        else:
            if len(roi_size) == 2:
                self.roi_size = np.array(roi_size, dtype=int)
            else:
                raise Exception(f'given roi_size is not valid. Must be list or tuple of length 2 or int')

    
    def show_points(self, video, roi_size=None):
        """
        Show points to be analyzed, together with ROI borders.
        """
        if roi_size is None:
            if hasattr(self, 'roi_size'):
                roi_size = self.roi_size

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.imshow(video.mraw[0].astype(float), cmap='gray')
        ax.scatter(video.points[:, 1],
                   video.points[:, 0], marker='.', color='r')

        if roi_size is not None:
            for point in video.points:
                roi_border = patches.Rectangle((point - self.roi_size//2)[::-1], self.roi_size[1], self.roi_size[0],
                                               linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(roi_border)

        plt.grid(False)
        plt.show()


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
        'max_nfev': video.method.max_nfev, 
        'tol': video.method.tol, 
        'verbose': video.method.verbose, 
        'show_pbar': video.method.show_pbar
    }
    
    pool = Pool(processes=processes)
    results = [pool.apply_async(worker, args=(p, idi_kwargs, method_kwargs)) for p in points_split]
    pool.close()
    pool.join()

    out = []
    for r in results:
        _r = r.get()
        for i in _r:
            out.append(i)
    
    return np.asarray(out)


def worker(points, idi_kwargs, method_kwargs):
    """
    A function that is called when for each job in multiprocessing.
    """
    _video = pyidi.pyIDI(**idi_kwargs)
    _video.set_method(LucasKanadeSc)
    _video.method.configure(**method_kwargs)
    _video.set_points(points)
    
    return _video.get_displacements(verbose=0)












