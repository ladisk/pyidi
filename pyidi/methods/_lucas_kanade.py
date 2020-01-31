import numpy as np
import time
import datetime

import scipy.signal
from scipy.linalg import lu_factor, lu_solve
from scipy.interpolate import RectBivariateSpline
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from multiprocessing import Pool

from atpbar import atpbar
import mantichora

from psutil import cpu_count
from .. import pyidi
from .. import tools

from .idi_method import IDIMethod


class LucasKanade(IDIMethod):
    """
    Translation identification based on the Lucas-Kanade method using least-squares
    iterative optimization with the Zero Normalized Cross Correlation optimization
    criterium.
    """  
    def configure(
        self, roi_size=(9, 9), pad=2, max_nfev=20, 
        tol=1e-8, int_order=3, verbose=1, show_pbar=True, 
        processes=1, pbar_type='tqdm', multi_type='multiprocessing'
    ):
        """
        Displacement identification based on Lucas-Kanade method,
        using iterative least squares optimization of translatory transformation
        parameters to determine image ROI translations.
        
        :param video: parent object
        :type video: object
        :param roi_size: (h, w) height and width of the region of interest.
            ROI dimensions should be odd numbers. Defaults to (9, 9)
        :type roi_size: tuple, list, optional
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
        :param processes: number of processes to run with `multiprocessing`
        :type processes: int, optional, defaults to 1.
        :param pbar_type: type of the progress bar ('tqdm' or 'atpbar'), defaults to 'tqdm'
        :type pbar_type: str, optional
        :param multi_type: type of multiprocessing used ('multiprocessing' or 'mantichora'), defaults to 'multiprocessing'
        :type multi_type: str, optional
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


        if self.processes != 1:
            self.displacements = multi(video, self.processes)
            # return?

        else:            
            self.image_size = video.mraw.shape[-2:]

            self.displacements = np.zeros((video.points.shape[0], video.N, 2))
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
            for i in self._pbar_range(1, video.mraw.shape[0]):

                # Iterate over points.
                for p, point in enumerate(video.points):
                    
                    # start optimization with previous optimal parameter values
                    d_init = np.round(self.displacements[p, i-1, :]).astype(int)

                    yslice, xslice = self._padded_slice(point+d_init, self.roi_size, 0)
                    G = video.mraw[i, yslice, xslice]
                    displacements = self.optimize_translations(
                        G=G, 
                        F_spline=self.interpolation_splines[p], 
                        maxiter=self.max_nfev,
                        tol=self.tol
                        ) # input difference bwtween d_init and last d as d_subpixel_init??

                    self.displacements[p, i, :] = displacements + d_init
                    
            if self.verbose:
                full_time = time.time() - start_time
                if full_time > 60:
                    full_time_m = full_time//60
                    full_time_s = full_time%60
                    print(f'Time to complete: {full_time_m:.0f} min, {full_time_s:.1f} s')
                else:
                    print(f'Time to complete: {full_time:.1f} s')


    def optimize_translations(self, G, F_spline, maxiter, tol, d_subpixel_init=(0, 0)):
        """
        Determine the optimal translation parameters to align the current
        image subset `G` with the interpolated reference image subset `F`.
        
        :param G: the current image subset.
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
        Gy, Gx = np.gradient(G.astype(np.float64), edge_order=2)
        Gx2 = np.sum(Gx**2)
        Gy2 = np.sum(Gy**2)
        GxGy = np.sum(Gx * Gy)

        A_inv = np.linalg.inv(
            np.array([[GxGy, Gx2],  # switched columns, to reverse variable order to (dy, dx)
                      [Gy2, GxGy]])
                      )

        # initialize values
        error = 1.
        displacement = np.array(d_subpixel_init, dtype=np.float64)
        delta = displacement.copy()

        # optimization loop
        for i in range(maxiter):
            y_f = np.arange(self.roi_size[0], dtype=np.float64) + displacement[0]
            x_f = np.arange(self.roi_size[1], dtype=np.float64) + displacement[1]
            F = F_spline(y_f, x_f)

            F_G = G - F
            b = np.array([np.sum(Gx*F_G),
                          np.sum(Gy*F_G)
                ])
            delta = np.dot(A_inv, b)

            error = np.linalg.norm(delta)
            displacement += delta
            if error < tol:
                return -displacement # roles of F and G are switched

        # max_iter wa reached before the convergence criterium
        return -displacement


    def _padded_slice(self, point, roi_size, pad=None):
        '''
        Returns a slice that crops an image around a given `point` center, 
        `roi_size` and `pad` size.

        :param point: The center point coordiante of the desired ROI.
        :type point: array_like of size 2, (y, x)
        :param roi_size: Size of desired cropped image (y, x).
        type roi_size: array_like of size 2, (h, w)
        :param pad: Pad border size in pixels. If None, the video.pad
            attribute is read.
        :type pad: int, optional, defaults to None
        :return crop_slice: tuple (yslice, xslice) to use for image slicing.
        '''

        if pad is None:
            pad = self.pad
        y, x = np.array(point).astype(int)
        h, w = np.array(roi_size).astype(int)

        # CLIP ON EDGES!
        # y_range = np.array([y-h//2-pad, y+h//2+pad+1], dtype=int)
        # x_range = np.array([x-w//2-pad, x+w//2+pad+1], dtype=int)

        yslice = slice(y-h//2-pad, y+h//2+pad+1)
        xslice = slice(x-w//2-pad, x+w//2+pad+1)
        return yslice, xslice


    def _pbar_range(self, *args, **kwargs):
        """
        Set progress bar range or normal range.
        """
        if self.show_pbar:
            if self.pbar_type == 'tqdm':
                return tqdm(range(*args, **kwargs), ncols=100, leave=True)
            elif self.pbar_type == 'atpbar':
                return atpbar(range(*args, **kwargs), name=f'{self.video.points.shape[0]} points')
        else:
            return range(*args, **kwargs)


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
        f = video.mraw[0].copy().astype(float)
        splines = []
        for point in video.points:
            yslice, xslice = self._padded_slice(point, self.roi_size, pad)

            # debug
            spl = RectBivariateSpline(
               x=np.arange(-pad, self.roi_size[0]+pad),
               y=np.arange(-pad, self.roi_size[1]+pad),
               z=f[yslice, xslice],
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
        ax.imshow(video.mraw[0].astype(float), cmap=cmap)
        ax.scatter(video.points[:, 1],
                   video.points[:, 0], marker='.', color=color)

        for point in video.points:
            roi_border = patches.Rectangle((point - self.roi_size//2)[::-1], self.roi_size[1], self.roi_size[0],
                                            linewidth=1, edgecolor=color, facecolor='none')
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
        'show_pbar': video.method.show_pbar,
        'int_order': video.method.int_order,
        'pbar_type': video.method.pbar_type,
    }
    if video.method.pbar_type == 'atpbar':
        print(f'Computation start: {datetime.datetime.now()}')

    if video.method.multi_type == 'multiprocessing':
        if method_kwargs['pbar_type'] == 'atpbar':
            method_kwargs['pbar_type'] = 'tqdm'
            print(f'!!! WARNING: "atpbar" pbar_type was used with "multiprocessing". This is not supported. Changed pbar_type to "tqdm"')

        pool = Pool(processes=processes)
        results = [pool.apply_async(worker, args=(p, idi_kwargs, method_kwargs)) for p in points_split]
        pool.close()
        pool.join()

        out = []
        for r in results:
            _r = r.get()
            for i in _r:

                out.append(i)
    
    elif video.method.multi_type == 'mantichora':
        with mantichora.mantichora(nworkers=processes) as mcore:
            for p in points_split:
                mcore.run(worker, p, idi_kwargs, method_kwargs)
            returns = mcore.returns()
        
        out = []
        for r in returns:
            for i in r:
                out.append(i)
    
    return np.asarray(out)


def worker(points, idi_kwargs, method_kwargs):
    """
    A function that is called when for each job in multiprocessing.
    """
    _video = pyidi.pyIDI(**idi_kwargs)
    _video.set_method(LucasKanade)
    _video.method.configure(**method_kwargs)
    _video.set_points(points)
    
    return _video.get_displacements(verbose=0)