import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from multiprocessing import Pool
from tqdm import tqdm

class RegularROIGrid:
    """
    Automatic ROI grid generation.
    """
    def __init__(self, video, roi_size=(7, 7), noverlap=0, sssig_filter=None, verbose=1):
        """
        
        :param video: parent object of video
        :type video: object
        :param roi_size: Size of the region of interest (y, x), defaults to (7, 7)
        :type roi_size: tuple, list, optional
        :param noverlap: number of pixels that overlap between neighbouring ROIs
        :type noverlap: int, optional
        :param sssig_filter: minimum value of SSSIG that the roi must have, defaults to None
        :type sssig_filter: None, float, optional
        :param verbose: Show text, defaults to 1
        :type verbose: int, optional
        """
        self.roi_size = roi_size
        self.verbose = verbose
        if noverlap > np.min(np.asarray(self.roi_size)//2):
            print('!!!! WARNING: "noverlap" should be smaller than half of "roi_size"')

        self.noverlap = int(noverlap)

        self.sssig_filter = sssig_filter
        self.cent_dist_0 = self.roi_size[0] - self.noverlap
        self.cent_dist_1 = self.roi_size[1] - self.noverlap

        self.image = video.mraw[0]

        # Tkinter root and matplotlib figure
        root = tk.Tk()
        root.title('Pick points')
        fig = Figure(figsize=(15, 7))
        ax = fig.add_subplot(111)
        ax.grid(False)
        ax.imshow(self.image, cmap='gray')
        plt.show()

        # Embed figure in tkinter winodw
        canvas = FigureCanvasTkAgg(fig, root)
        canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
        NavigationToolbar2Tk(canvas, root)

        # Initiate polygon
        self.polygon = [[], []]
        line, = ax.plot(self.polygon[1], self.polygon[0], 'r.-')

        if self.verbose:
            print('SHIFT + LEFT mouse button to pick a pole.\nRIGHT mouse button to erase the last pick.')

        self.shift_is_held = False

        def on_key_press(event):
            """Function triggered on key press (shift)."""
            if event.key == 'shift':
                self.shift_is_held = True

        def on_key_release(event):
            """Function triggered on key release (shift)."""
            if event.key == 'shift':
                self.shift_is_held = False

        def onclick(event):
            if event.button == 1 and self.shift_is_held:
                if event.xdata is not None and event.ydata is not None:
                    self.polygon[1].append(int(np.round(event.xdata)))
                    self.polygon[0].append(int(np.round(event.ydata)))
                    if self.verbose:
                        print(f'y: {np.round(event.ydata):5.0f}, x: {np.round(event.xdata):5.0f}')

            elif event.button == 3 and self.shift_is_held:
                if self.verbose:
                    print('Deleted the last point...')
                del self.polygon[1][-1]
                del self.polygon[0][-1]

            line.set_xdata(self.polygon[1])
            line.set_ydata(self.polygon[0])
            fig.canvas.draw()

        def handle_close(event):
            """On closing."""
            self.polygon = np.asarray(self.polygon).T
            if self.verbose:
                for i, point in enumerate(self.polygon):
                    print(f'{i+1}. point: x ={point[1]:5.0f}, y ={point[0]:5.0f}')
            
            self.points, self.roi_quality = self.get_roi_grid()
            self.ROI_filter_SSSIG()


        # Connecting functions to event manager
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        fig.canvas.mpl_connect('key_release_event', on_key_release)
        fig.canvas.mpl_connect('button_press_event', onclick)
        # on closing the figure
        fig.canvas.mpl_connect('close_event', handle_close)

        root.mainloop()


    def get_roi_grid(self):
        """Get the points inside the polygon and SSSIG values that correspond to the ROI.
        """
        min_0 = np.min(self.polygon[:, 0])
        max_0 = np.max(self.polygon[:, 0])
        min_1 = np.min(self.polygon[:, 1])
        max_1 = np.max(self.polygon[:, 1])
        
        x = self.polygon[:, 1]
        y = self.polygon[:, 0]

        _polygon = np.asarray([x, y]).T
        
        sp_0 = []
        sp_1 = []
        roi_quality = []
        for p0 in range(min_0+1, max_0):
            for p1 in range(min_1+1, max_1):
                if inside_polygon(p0, p1, self.polygon):
                    if not sp_0 or (np.abs(p0 - sp_0[-1]) >= self.cent_dist_0 or np.abs(p1 - sp_1[-1] >= self.cent_dist_1)):
                        sp_0.append(p0)
                        sp_1.append(p1)
                        roi_quality.append(self._sssig(p0, p1))

        return np.column_stack((sp_0, sp_1)), np.asarray(roi_quality)


    def _sssig(self, p0, p1):
        """Determine SSSIG value for a ROI.
        """
        coor_0 = p0 + np.arange(-self.roi_size[0]//2, self.roi_size[0]//2)
        coor_1 = p1 + np.arange(-self.roi_size[1]//2, self.roi_size[1]//2)
        m0, m1 = np.meshgrid(coor_0, coor_1)
        g0, g1 = np.gradient(self.image[m0, m1])

        val = np.sqrt(np.sum(g0**2) + np.sum(g1**2)) / (self.roi_size[0]*self.roi_size[1])
        return val


    def ROI_filter_SSSIG(self):
        """Filter the points according to SSSIG value.
        """
        if self.sssig_filter is not None:
            f = float(self.sssig_filter)

            sel = (self.roi_quality > f)
            self.roi_quality = self.roi_quality[sel]
            self.points = self.points[sel]


    def show_grid(self, highlight=None, show_rois=True):
        """Show the image with ROIs and points.
        """
        fig, ax = plt.subplots()
        ax.imshow(self.image, 'gray')

        polygon = np.concatenate((self.polygon, self.polygon[0:1]), axis=0)
        ax.plot(polygon[:, 1], polygon[:, 0], 'y')
        for i, p in enumerate(self.points):
            ax.plot(p[1], p[0], 'g.')
            if show_rois:
                ax.add_patch(patches.Rectangle(
                    (p[1]-self.roi_size[1]//2, p[0]-self.roi_size[0]//2), 
                    self.roi_size[1], self.roi_size[0], 
                    fill=False, color='g'))

        if highlight is not None:
            hL = int(highlight)
            ax.plot(self.points[hL, 1], self.points[hL, 0], 'ro')
            ax.add_patch(patches.Rectangle(
                (self.points[hL, 1]-self.roi_size[1]//2, self.points[hL, 0]-self.roi_size[0]//2), 
                self.roi_size[1], self.roi_size[0], 
                fill=False, color='r', linewidth=2))


class ManualROI:
    """Manual ROI selection."""

    def __init__(self, video, roi_size, single=False, verbose=0):
        """Manually select region of interest.
        
        :param video: parent object
        :type video: object
        :param roi_size: size of region of interest (dy, dx)
        :type roi_size: tuple, list
        :param single: if True, ONLY ONE ROI can be selected, defaults to False
        :type single: bool, optional
        :param verbose: Show text, defaults to 0
        :type verbose: int, optional
        """
        self.roi_size = roi_size
        self.image = video.mraw[0]
        self.verbose = verbose

        # Tkinter root and matplotlib figure
        root = tk.Tk()
        root.title('Pick point')
        fig = Figure(figsize=(15, 7))
        ax = fig.add_subplot(111)
        ax.grid(False)
        ax.imshow(self.image, cmap='gray')
        plt.show()

        # Embed figure in tkinter winodw
        canvas = FigureCanvasTkAgg(fig, root)
        canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
        NavigationToolbar2Tk(canvas, root)

        if self.verbose:
            print('SHIFT + LEFT mouse button to pick a pole.\nRIGHT mouse button to erase the last pick.')


        self.point = [[], []]
        line, = ax.plot(self.point[1], self.point[0], 'r.')
        
        self.rectangles = []
        self.rectangles.append(patches.Rectangle((0, 0), 10, 10, fill=False, alpha=0))
        ax.add_patch(self.rectangles[-1])        
        
        self.shift_is_held = False
        def on_key_press(event):
            """Function triggered on key press (shift)."""
            if event.key == 'shift':
                self.shift_is_held = True

        def on_key_release(event):
            """Function triggered on key release (shift)."""
            if event.key == 'shift':
                self.shift_is_held = False

        def onclick(event):
            if event.button == 1 and self.shift_is_held:
                if event.xdata is not None and event.ydata is not None:
                    if single:
                        self.point[0] = [int(np.round(event.ydata))]
                        self.point[1] = [int(np.round(event.xdata))]
                    else:
                        self.point[0].append(int(np.round(event.ydata)))
                        self.point[1].append(int(np.round(event.xdata)))
                    if self.verbose:
                        print(f'y: {np.round(event.ydata):5.0f}, x: {np.round(event.xdata):5.0f}')

            elif event.button == 3 and self.shift_is_held and not single:
                if self.verbose:
                    print('Deleted the last point...')
                del self.point[1][-1]
                del self.point[0][-1]
                del self.rectangles[-1]

            line.set_xdata(self.point[1])
            line.set_ydata(self.point[0])
            
            if self.point[0]:
                [p.remove() for p in reversed(ax.patches)]
                self.rectangles = []
                for i, (p0, p1) in enumerate(zip(self.point[0], self.point[1])):
                    self.rectangles.append(patches.Rectangle((p1 - self.roi_size[1]//2, p0 - self.roi_size[0]//2), 
                                                    self.roi_size[1], self.roi_size[0], fill=False, color='r', linewidth=2))
                    ax.add_patch(self.rectangles[-1])

            fig.canvas.draw()

        def handle_close(event):
            """On closing."""
            self.points = np.asarray(self.point).T
            if self.verbose:
                for i, point in enumerate(self.polygon):
                    print(f'{i+1}. point: x ={point[1]:5.0f}, y ={point[0]:5.0f}')

        # Connecting functions to event manager
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        fig.canvas.mpl_connect('key_release_event', on_key_release)
        fig.canvas.mpl_connect('button_press_event', onclick)
        # on closing the figure
        fig.canvas.mpl_connect('close_event', handle_close)

        root.mainloop()
    

class GridOfROI:
    """
    Automatic simple ROI grid generation.

    Different from RegularROIGrid in that it gets a regular grid and only
    then checks if all points are inside polygon. This yields a more regular
    and full grid. Does not contain sssig filter.
    """
    def __init__(self, video, roi_size=(7, 7), noverlap=0, verbose=0):
        """
        
        :param video: parent object of video
        :type video: object
        :param roi_size: Size of the region of interest (y, x), defaults to (7, 7)
        :type roi_size: tuple, list, optional
        :param noverlap: number of pixels that overlap between neighbouring ROIs
        :type noverlap: int, optional
        :param sssig_filter: minimum value of SSSIG that the roi must have, defaults to None
        :type sssig_filter: None, float, optional
        :param verbose: Show text, defaults to 1
        :type verbose: int, optional
        """
        self.roi_size = roi_size
        self.verbose = verbose
        if noverlap > np.min(np.asarray(self.roi_size)//2):
            print('!!!! WARNING: "noverlap" should be smaller than half of "roi_size"')

        self.noverlap = int(noverlap)

        self.cent_dist_0 = self.roi_size[0] - self.noverlap
        self.cent_dist_1 = self.roi_size[1] - self.noverlap

        self.image = video.mraw[0]

        # Tkinter root and matplotlib figure
        root = tk.Tk()
        root.title('Pick points')
        fig = Figure(figsize=(15, 7))
        ax = fig.add_subplot(111)
        ax.grid(False)
        ax.imshow(self.image, cmap='gray')
        plt.show()

        # Embed figure in tkinter winodw
        canvas = FigureCanvasTkAgg(fig, root)
        canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
        NavigationToolbar2Tk(canvas, root)

        # Initiate polygon
        self.polygon = [[], []]
        line, = ax.plot(self.polygon[1], self.polygon[0], 'r.-')

        if self.verbose:
            print('SHIFT + LEFT mouse button to pick a pole.\nRIGHT mouse button to erase the last pick.')

        self.shift_is_held = False

        def on_key_press(event):
            """Function triggered on key press (shift)."""
            if event.key == 'shift':
                self.shift_is_held = True

        def on_key_release(event):
            """Function triggered on key release (shift)."""
            if event.key == 'shift':
                self.shift_is_held = False

        def onclick(event):
            if event.button == 1 and self.shift_is_held:
                if event.xdata is not None and event.ydata is not None:
                    self.polygon[1].append(int(np.round(event.xdata)))
                    self.polygon[0].append(int(np.round(event.ydata)))
                    if self.verbose:
                        print(f'y: {np.round(event.ydata):5.0f}, x: {np.round(event.xdata):5.0f}')

            elif event.button == 3 and self.shift_is_held:
                if self.verbose:
                    print('Deleted the last point...')
                del self.polygon[1][-1]
                del self.polygon[0][-1]

            line.set_xdata(self.polygon[1])
            line.set_ydata(self.polygon[0])
            fig.canvas.draw()

        def handle_close(event):
            """On closing."""
            self.polygon = np.asarray(self.polygon).T
            if self.verbose:
                for i, point in enumerate(self.polygon):
                    print(f'{i+1}. point: x ={point[1]:5.0f}, y ={point[0]:5.0f}')
            
            self.points = self.get_roi_grid()

        # Connecting functions to event manager
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        fig.canvas.mpl_connect('key_release_event', on_key_release)
        fig.canvas.mpl_connect('button_press_event', onclick)
        # on closing the figure
        fig.canvas.mpl_connect('close_event', handle_close)

        root.mainloop()
    
    def get_roi_grid(self):
        points = self.polygon
        
        low_0 = np.min(points[:, 0])
        high_0 = np.max(points[:, 0])
        low_1 = np.min(points[:, 1])
        high_1 = np.max(points[:, 1])
        
        rois = []
        for i in range(low_0+self.cent_dist_0, high_0-self.cent_dist_0, self.cent_dist_0):
            for j in range(low_1+self.cent_dist_1, high_1-self.cent_dist_1, self.cent_dist_1):
                if inside_polygon(i, j, self.polygon):
                    rois.append([i, j])
        return np.asarray(rois)



def inside_polygon(x, y, points):
    """
    Return True if a coordinate (x, y) is inside a polygon defined by
    a list of verticies [(x1, y1), (x2, x2), ... , (xN, yN)].

    Reference: http://www.ariel.com.au/a/python-point-int-poly.html
    """
    n = len(points)
    inside = False
    p1x, p1y = points[0]
    for i in range(1, n + 1):
        p2x, p2y = points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / \
                            (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def update_docstring(target_method, doc_method=None, delimiter='---', added_doc=''):
    """
    Update the docstring in target_method with the docstring from doc_method.
    
    :param target_method: The method that waits for the docstring
    :type target_method: method
    :param doc_method: The method that holds the desired docstring
    :type doc_method: method
    :param delimiter: insert the desired docstring between two delimiters, defaults to '---'
    :type delimiter: str, optional
    """
    docstring = target_method.__doc__.split(delimiter)
    leading_spaces = len(docstring[1].replace('\n', '')) - len(docstring[1].replace('\n', '').lstrip(' '))
    
    if doc_method is not None:
        if doc_method.__doc__:
            docstring[1] = doc_method.__doc__
        else:
            docstring[1] = '\n' + ' '*leading_spaces + \
                'The selected method does not have a docstring.\n'
    else:
        docstring[1] = added_doc.replace('\n', '\n' + ' '*leading_spaces)

    target_method.__func__.__doc__ = delimiter.join(docstring)


def func_4_multi(video, points):
    """
    A function that is called when for each job in multiprocessing.
    """
    video.set_points(points)
    return video.get_displacements(verbose=0)


def multi(video, points, processes=2):
    """
    Compute the displacements using multiprocessing.
    
    :param video: The pyIDI object with defined method
    :type video: object
    :param points: 2d array with point indices
    :type points: ndarray
    :param processes: number of processes, defaults to 2
    :type processes: int, optional
    :return: displacement array (3d array)
    :rtype: ndarray
    """
    def update(a):
        pbar.update(1)

    pbar = tqdm(total=points.shape[0])

    pool = Pool(processes=processes)
    results = [pool.apply_async(func_4_multi, args=(video, points[i].reshape(1, 2)), callback=update) for i in range(points.shape[0])]
    pool.close()
    pool.join()
    pbar.close()

    out = np.array([r.get() for r in results]).reshape(points.shape[0], -1, 2)
    
    return out